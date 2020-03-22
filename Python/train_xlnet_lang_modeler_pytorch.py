import transformers
import torch
import wandb
import time
import os
import argparse
import json
from chunked_text_dataloader import ChunkedTextDataset
from tqdm import tqdm

fp16 = True
if fp16:
    from apex import amp


class Trainer:
    def __init__(self, model, chunked_model_config, train_dataloader, val_dataloader, optimizer, scheduler, device="cuda", is_fp16=False, desired_batch_sz=64):
        self.model = model
        self.chunked_model_config = chunked_model_config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.is_fp16 = is_fp16
        self.desired_batch_sz = desired_batch_sz
        assert(self.desired_batch_sz % self.chunked_model_config["batch_size"] == 0)

        self.preprocess_times = []
        self.forward_times = []
        self.backward_times = []
        self.opt_times = []

    def clear_timers(self):
        self.forward_times.clear()
        self.backward_times.clear()
        self.opt_times.clear()

    def save_model(self, _chkpt_name):
        # Save the model
        _output_dir = os.path.join(
            self.chunked_model_config["output_dir"],
            "xlnet_trainer_checkpoints",
            _chkpt_name,
        )

        if not os.path.exists(_output_dir):
            os.makedirs(_output_dir)

        # Save configuration options specific to this run.
        with open(
            os.path.join(_output_dir, "chunk_config.json"), "w"
        ) as _chunk_config_file:
            json.dump(self.chunked_model_config, _chunk_config_file)

        # Save processing times.
        _times = {
            "preprocess": self.preprocess_times,
            "forward": self.forward_times,
            "backward": self.backward_times,
            "opt": self.opt_times,
        }
        with open(
            os.path.join(_output_dir, "processing_times.pt"), "w"
        ) as _processing_times_file:
            json.dump(_times, _processing_times_file)

        self.model = (
            self.model if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training
        self.model.save_pretrained(_output_dir)
        print("Save completed. %s" % (_output_dir))


    def train_epoch(self):
        _logging_steps = 5
        _steps_till_save = 2000
        _steps_till_validate = 2000

        self.clear_timers()

        _epoch_iterator = tqdm(self.train_dataloader, desc="Train Iteration")
        _steps, _optimizer_steps = 0, 0
        _tr_loss, _logging_loss = 0, 0
        _chunks = 0
        _accuracy_accum, _accuracy_last = 0, 0
        self.model.train()

        # This controls how many batches are required per optimizer step.
        _batches_required_for_desired_sz = int(self.desired_batch_sz / self.chunked_model_config["batch_size"])
        _cur_step = 0

        __s = time.time()
        for _step, _batch in enumerate(_epoch_iterator):
            self.preprocess_times.append(time.time() - __s)

            _mems = None
            _loss = None
            _chunk_loss_schedule = []
            _num_chunks = len(_batch["input_ids"])
            _chunks += _num_chunks
            for _masked_input_ids, _attention_masks, _labels in zip(
                _batch["input_ids_masked"], _batch["attention_masks"], _batch["labels"]
            ):
                # Forward
                _inputs = {
                    "input_ids": _masked_input_ids.to(self.device),
                    "attention_mask": _attention_masks.to(self.device),
                    "labels": _labels.to(self.device),
                }
                if _mems is not None:
                    _inputs["mems"] = _mems

                __s = time.time()
                _loss, _logits, _mems = self.model.forward(**_inputs)
                self.forward_times.append(time.time() - __s)

                _chunk_loss_schedule.append(_loss.item())

                # Backwards
                __s = time.time()
                if fp16:
                    with amp.scale_loss(_loss, self.optimizer) as _scaled_loss:
                        _loss.backward()
                        backward_time = time.time() - __s
                else:
                    _loss.backward()
                    backward_time = time.time() - __s
                if self.is_fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), 1)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.backward_times.append(backward_time)

            # Update weights after all chunks have been processed at an interval to fulfill desired_batch_sz
            _cur_step += 1
            if _cur_step % _batches_required_for_desired_sz == 0:
                __s = time.time()
                self.optimizer.step()
                self.opt_times.append(time.time() - __s)
                self.scheduler.step()
                self.model.zero_grad()
                _optimizer_steps += 1

            # Always accumulate loss across the last chunk, where it should be lowest. That's the goal of this model.
            _tr_loss += _loss.item()
            if _steps % _logging_steps == 0:
                _loss_scalar = (_tr_loss - _logging_loss) / _logging_steps
                _logging_loss = _tr_loss
                _logs = {}
                _logs["avg_chunks"] = _chunks / _logging_steps
                _chunks = 0
                _logs["loss"] = _loss_scalar
                _logs["learning_rate"] = self.scheduler.get_lr()[0]
                _logs["optimizer_steps"] = _optimizer_steps
                if do_wandb:
                    # wandb can fail if the network connection goes down. this shouldn't take down training.
                    try:
                        wandb.log(_logs)
                    except:
                        print(_logs)
                else:
                    print(_logs)

            if _steps != 0 and _steps % _steps_till_save == 0:
                self.save_model("chkpt_%i" % (_steps))
            if _steps != 0 and _steps % _steps_till_validate == 0:
                self.validate()

            _steps += 1
            # Record time so we see how long it takes to fetch a batch.
            __s = time.time()


    def validate(self):
        _epoch_iterator = tqdm(self.val_dataloader, desc="Validation Iteration")
        _actual_steps = 0
        _total_loss = 0

        with torch.no_grad():
            for _step, _batch in enumerate(_epoch_iterator):
                # Skip 1 in 10 steps, because the validator just takes too long otherwise. It's not as easy as just cutting
                # down the dataset, either, since we run into chunk/batch size mismatches then.
                if _step % 10 != 0:
                    continue
                _mems = None
                _loss = None
                for _masked_input_ids, _attention_masks, _labels in zip(
                    _batch["input_ids_masked"], _batch["attention_masks"], _batch["labels"]
                ):
                    # Forward
                    _inputs = {
                        "input_ids": _masked_input_ids.to(self.device),
                        "attention_mask": _attention_masks.to(self.device),
                        "labels": _labels.to(self.device),
                    }
                    if _mems is not None:
                        _inputs["mems"] = _mems

                    _loss, _logits, _mems = self.model.forward(**_inputs)

                # Always accumulate loss across the last chunk, where it should be lowest. That's the goal of this model.
                _actual_steps += 1
                _total_loss += _loss.item()

            _logs = {}
            _val_loss = _total_loss / _actual_steps
            _logs["val_loss"] = _val_loss
            if do_wandb:
                wandb.log(_logs)
            print("Validation loss: " + str(_val_loss))


if __name__ == "__main__":
    run_name = input("Enter a name for this run..")

    # Process command line flags
    parser = argparse.ArgumentParser(
        description="Train an auto-regressive transformer model."
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="nonint-transformers-torch",
        help="Project name for wandb",
    )
    parser.add_argument("--batch_sz", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--seq_sz", type=int, default=256, help="Sequence size to be fed into the model"
    )
    parser.add_argument(
        "--max_predict_sz",
        type=int,
        default=32,
        help="Max sequence size of the predicted sequence",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="xlnet-base-cased",
        help="Transformers pre-trained model to start with",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to train dataset against.",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Where to find train.pt and val.pt datasets.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="PyTorch device name to run on."
    )
    parser.add_argument(
        "--start_lr", type=float, default=2e-5, help="Learning rate to start at."
    )
    parser.add_argument(
        "--output_dir", type=str, default=".", help="Directory where checkpoints saves will be made."
    )
    args = parser.parse_args()

    project_name = args.project_name
    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_sz
    input_folder = args.input_folder
    torch_device_name = args.device
    start_lr = args.start_lr

    chunked_model_config = {
        "name": run_name,
        "max_seq_len": args.seq_sz,
        "model_name": args.model_name,
        "predict_len": args.max_predict_sz,
        "batch_size": batch_size,
        "starting_lr": start_lr,
        "output_dir": args.output_dir,
        "target_mask_percent": 0.3,
        "target_mask_cluster_count": 3,
        "text_mask_percentage": 0.1,
        "force_max_len_gen": False,
        "mem_len": 1024,
    }

    tokenizer = transformers.XLNetTokenizer.from_pretrained(
        chunked_model_config["model_name"]
    )

    # Get the datasets
    print("*** Loading data.. ***")
    train_set = ChunkedTextDataset(
        os.path.join(input_folder, "train.pt"),
        tokenizer,
        chunked_model_config["max_seq_len"],
        chunked_model_config["predict_len"],
        mask_target_percentage=chunked_model_config["target_mask_percent"],
        mask_all_percentage=chunked_model_config["text_mask_percentage"],
        pad_left=True,
        force_max_len_gen=chunked_model_config["force_max_len_gen"],
        target_mask_cluster_count=chunked_model_config["target_mask_cluster_count"],
        cluster_easing=False
    )
    val_set = ChunkedTextDataset(
        os.path.join(input_folder, "val.pt"),
        tokenizer,
        chunked_model_config["max_seq_len"],
        chunked_model_config["predict_len"],
        mask_target_percentage=chunked_model_config["target_mask_percent"],
        mask_all_percentage=chunked_model_config["text_mask_percentage"],
        pad_left=True,
        force_max_len_gen=chunked_model_config["force_max_len_gen"]
    )
    train_loader = train_set.get_dataloader(batch_size, num_workers=0)
    val_loader = val_set.get_dataloader(batch_size, num_workers=0, random=False)

    # Initialize w&b logger
    do_wandb = True
    if do_wandb:
        wandb.init(project=project_name, name=run_name, config=chunked_model_config)
        # There's something bugged about this, but it doesnt really seem to do much anyways. Apparently it enables some
        # sort of gradient exploration map.
        # wandb.watch(model)

    # Load model
    print("*** Loading model.. ***")
    config = transformers.XLNetConfig.from_pretrained(
        chunked_model_config["model_name"]
    )
    config.mem_len = chunked_model_config["mem_len"]
    model = transformers.XLNetLMHeadModel.from_pretrained(
        chunked_model_config["model_name"], config=config
    )
    device = torch.device(torch_device_name)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=start_lr, eps=1e-8)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        # '5' a tad bit higher than the average number of chunks,
        #  each of which will get a train step.
        num_training_steps=epochs * len(train_set) * 5 / batch_size,
    )

    # Shift model to device & enable fp16 if applicable.
    model.to(device)
    if fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    print("*** Running training ***")
    trainer = Trainer(model, chunked_model_config, train_loader, val_loader, optimizer, scheduler, device, fp16)
    model.zero_grad()
    for _ in range(epochs):
        trainer.train_epoch()
        # Slowly increase the mask percentage per epoch to make the model have to work harder.
        train_set.mask_target_percentage += 0.1

    trainer.validate()
    trainer.save_model("final")

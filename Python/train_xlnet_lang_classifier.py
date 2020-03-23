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

preprocess_times = []
forward_times = []
backward_times = []
opt_times = []


def clear_timers():
    forward_times.clear()
    backward_times.clear()
    opt_times.clear()


def save_model(_model, _chkpt_name, _chunked_model_config):
    # Save the model
    _output_dir = os.path.join(
        "c:/Users/jbetk/Documents/data/ml/saved_models",
        "xlnet_classification",
        _chkpt_name,
    )

    if not os.path.exists(_output_dir):
        os.makedirs(_output_dir)

    # Save configuration options specific to this run.
    with open(
        os.path.join(_output_dir, "chunk_config.json"), "w"
    ) as _chunk_config_file:
        json.dump(_chunked_model_config, _chunk_config_file)

    # Save processing times.
    _times = {
        "preprocess": preprocess_times,
        "forward": forward_times,
        "backward": backward_times,
        "opt": opt_times,
    }
    with open(
        os.path.join(_output_dir, "processing_times.pt"), "w"
    ) as _processing_times_file:
        json.dump(_times, _processing_times_file)

    _model_to_save = (
        _model.module if hasattr(_model, "module") else _model
    )  # Take care of distributed/parallel training
    _model_to_save.save_pretrained(_output_dir)
    print("Save completed. %s" % (_output_dir))


def train_epoch(
    _model, _optimizer, _scheduler, _device, _dataloader, _chunked_model_config, _fp16
):
    _logging_steps = 5
    _steps_till_save = 2000
    _steps_till_validate = 2000

    clear_timers()

    _epoch_iterator = tqdm(_dataloader, desc="Iteration")
    _steps = 0
    _tr_loss, _logging_loss = 0, 0
    _chunks = 0
    _accuracy_accum, _accuracy_last = 0, 0
    _model.train()

    __s = time.time()
    for _step, _batch in enumerate(_epoch_iterator):
        preprocess_times.append(time.time() - __s)

        _mems = None
        _loss = None
        _num_chunks = len(_batch["input_ids"])
        _chunks += _num_chunks
        for _input_ids, _token_type_ids, _attention_masks, _labels in zip(
            _batch["input_ids"],
            _batch["token_type_ids"],
            _batch["attention_masks"],
            _batch["classifiers"],
        ):
            # Forward
            _inputs = {
                "input_ids": _input_ids.to(_device),
                "token_type_ids": _token_type_ids.to(_device),
                "attention_mask": _attention_masks.to(_device),
                "labels": _labels.to(_device),
            }
            if _mems is not None:
                _inputs["mems"] = _mems

            __s = time.time()
            _loss, _logits, _mems = _model.forward(**_inputs)
            forward_times.append(time.time() - __s)

            # Backwards
            # Scale loss by the chunk size to give all sequences equal importance.
            _scaled_loss = _loss / _num_chunks
            __s = time.time()
            if fp16:
                with amp.scale_loss(_scaled_loss, _optimizer) as _scaled_loss:
                    _scaled_loss.backward()
                    backward_time = time.time() - __s
            else:
                _scaled_loss.backward()
                backward_time = time.time() - __s
            backward_times.append(backward_time)

            # Update weights after all chunks have been processed.
            if _fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(_optimizer), 1)
            else:
                torch.nn.utils.clip_grad_norm_(_model.parameters(), 1)
        __s = time.time()
        _optimizer.step()
        opt_times.append(time.time() - __s)
        _scheduler.step()
        _model.zero_grad()

        # Always accumulate loss across the last chunk, where it should be lowest. That's the goal of this model.
        _tr_loss += _loss.item()

        if _steps % _logging_steps == 0:
            _loss_scalar = (_tr_loss - _logging_loss) / _logging_steps
            _logging_loss = _tr_loss
            _logs = {}
            _logs["avg_chunks"] = _chunks / _logging_steps
            _chunks = 0
            _logs["loss"] = _loss_scalar
            _logs["learning_rate"] = _scheduler.get_lr()[0]
            if do_wandb:
                # wandb can fail if the network connection goes down. this shouldn't take down training.
                try:
                    wandb.log(_logs)
                except:
                    print(_logs)
            else:
                print(_logs)

        if _steps % _steps_till_save == 0:
            save_model(model, "chkpt_%i" % (_steps), _chunked_model_config)
        if _steps % _steps_till_validate == 0:
            validate(_model, _device)

        _steps += 1
        # Record time so we see how long it takes to fetch a batch.
        __s = time.time()


def validate(_model, _device):
    _epoch_iterator = tqdm(val_loader, desc="Validation Iteration")
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
            for _input_ids, _token_type_ids, _attention_masks, _labels in zip(
                _batch["input_ids"],
                _batch["token_type_ids"],
                _batch["attention_masks"],
                _batch["classifiers"],
            ):
                # Forward
                _inputs = {
                    "input_ids": _input_ids.to(_device),
                    "token_type_ids": _token_type_ids.to(_device),
                    "attention_mask": _attention_masks.to(_device),
                    "labels": _labels.to(_device),
                }
                if _mems is not None:
                    _inputs["mems"] = _mems

                _loss, _logits, _mems = _model.forward(**_inputs)

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
        description="Train an sequence classification XLNet transformer model."
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
        "target_mask_percent": 0.0,
        "target_mask_cluster_count": 1,
        "text_mask_percentage": 0.1,
        "force_max_len_gen": True,
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
        cluster_easing=False,
        includes_classification=True,
    )
    val_set = ChunkedTextDataset(
        os.path.join(input_folder, "val.pt"),
        tokenizer,
        chunked_model_config["max_seq_len"],
        chunked_model_config["predict_len"],
        mask_target_percentage=chunked_model_config["target_mask_percent"],
        mask_all_percentage=chunked_model_config["text_mask_percentage"],
        pad_left=True,
        force_max_len_gen=chunked_model_config["force_max_len_gen"],
        includes_classification=True,
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
    config.num_labels = 1  # Just straight-up MSE for this.
    model = transformers.XLNetForSequenceClassification.from_pretrained(
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
    model.zero_grad()
    for _ in range(epochs):
        train_epoch(
            model,
            optimizer,
            scheduler,
            device,
            train_loader,
            chunked_model_config,
            fp16,
        )

    validate(model, device)
    save_model(model, "final", chunked_model_config)

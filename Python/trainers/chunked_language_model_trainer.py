import torch
import wandb
import time
import os
import json
from tqdm import tqdm
from apex import amp

class ChunkedLMTrainer:
    def __init__(
        self,
        model,
        chunked_model_config,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        device="cuda",
        is_fp16=False,
        desired_batch_sz=64,
        do_wandb=False,
    ):
        self.model = model
        self.chunked_model_config = chunked_model_config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.is_fp16 = is_fp16
        self.do_wandb = do_wandb
        self.desired_batch_sz = desired_batch_sz
        assert self.desired_batch_sz % self.chunked_model_config["batch_size"] == 0

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

    def loop(self, _validate=False, _skip_batches=1):
        # How many steps per logging event.
        _logging_steps = 5
        # How many steps between checkpoint save and validation.
        _steps_till_save = 2000
        _steps_till_validate = 2000

        _dataloader = self.val_dataloader if _validate else self.train_dataloader
        _epoch_iterator = tqdm(
            _dataloader, desc="Val Iteration" if _validate else "Train Iteration")

        # Total batches processed and number of times optimizer.step() has been called.
        _steps, _optimizer_steps = 0, 0
        # Aggregate losses.
        _loss_sum, _logging_loss = 0, 0
        _chunks = 0
        self.clear_timers()

        # This controls how many batches are required per optimizer step.
        _batches_required_for_desired_sz = int(
            self.desired_batch_sz / self.chunked_model_config["batch_size"]
        )

        if _validate:
            torch.set_grad_enabled(False)
            model.eval()
        else:
            self.model.train()

        __s = time.time()
        for _step, _batch in enumerate(_epoch_iterator):
            # Skip batches if necessary. Usually set during validation to speed it up.
            if _step % _skip_batches != 0:
                continue

            self.preprocess_times.append(time.time() - __s)

            _mems = None
            _loss = None
            _num_chunks = len(_batch["input_ids"])
            _chunks += _num_chunks
            _chunk_counter = 0

            # Labels and target_mapping are not chunked.
            _labels = _batch["labels"]
            _target_mapping = _batch["target_mapping"]

            for _masked_input_ids, _attention_masks, _perm_mask in zip(
                _batch["input_ids"],
                _batch["attention_masks"],
                _batch["permutation_masks"]
            ):
                _is_last_chunk = _chunk_counter == (_num_chunks - 1)
                _chunk_counter += 1

                # Forward
                _inputs = {
                    "input_ids": _masked_input_ids.to(self.device),
                    "attention_mask": _attention_masks.to(self.device),
                    "perm_mask": _perm_mask.to(self.device),
                }
                if _mems is not None:
                    _inputs["mems"] = _mems

                __s = time.time()
                # Only compute gradients on the last forward() per-chunkset.
                if _is_last_chunk:
                    _inputs["target_mapping"] = _target_mapping.to(self.device)
                    _inputs["labels"] = _labels.to(self.device)
                    _loss, _logits, _mems = self.model.forward(**_inputs)
                else:
                    with torch.no_grad():
                        _logits, _mems = self.model.forward(**_inputs)
                self.forward_times.append(time.time() - __s)

                # Backwards
                # Only compute backwards on the last chunk per chunkset.
                if not _validate and _is_last_chunk:
                    __s = time.time()
                    if self.is_fp16:
                        with amp.scale_loss(_loss, self.optimizer) as _scaled_loss:
                            _loss.backward()
                            backward_time = time.time() - __s
                    else:
                        _loss.backward()
                        backward_time = time.time() - __s
                    if self.is_fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(self.optimizer), 1
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    self.backward_times.append(backward_time)

            if not _validate and _step % _batches_required_for_desired_sz == 0:
                # Update weights after all chunks have been processed at an interval to fulfill desired_batch_sz
                __s = time.time()
                self.optimizer.step()
                self.opt_times.append(time.time() - __s)
                self.scheduler.step()
                self.model.zero_grad()
                _optimizer_steps += 1

            # Always accumulate loss across the last chunk, where it should be lowest. That's the goal of this model.
            _loss_sum += _loss.item()
            if not _validate and _steps % _logging_steps == 0:
                _loss_scalar = (_loss_sum - _logging_loss) / _logging_steps
                _logging_loss = _loss_sum
                _logs = {
                    "avg_chunks": _chunks / _logging_steps,
                    "loss": _loss_scalar,
                    "learning_rate": self.scheduler.get_lr()[0],
                    "optimizer_steps": _optimizer_steps,
                }
                _chunks = 0
                if self.do_wandb:
                    # wandb can fail if the network connection goes down. this shouldn't take down training.
                    try:
                        wandb.log(_logs)
                    except:
                        print(_logs)
                else:
                    print(_logs)

            # The train loop automatically runs a validation loop and saves checkpoints.
            if not _validate:
                if _steps % _steps_till_save == 0:
                    self.save_model("chkpt_%i" % (_steps))
                if _steps % _steps_till_validate == 0:
                    self.loop(_validate=True, _skip_batches=10)

            _steps += 1
            # Record time so we see how long it takes to fetch a batch.
            __s = time.time()

        # Undo all the state changes needed for validation and perform validation logging.
        if _validate:
            torch.set_grad_enabled(True)
            model.train()

            _logs = {"val_loss": _loss_sum / _steps}
            if self.do_wandb:
                wandb.log(_logs)
            print("Validation loss: " + str(_logs["val_loss"]))

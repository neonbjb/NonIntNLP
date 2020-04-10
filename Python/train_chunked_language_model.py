import transformers
import torch
import wandb
import os
import argparse
from dataloaders.chunked_text_dataloader import ChunkedTextDataset
from trainers.chunked_language_model_trainer import ChunkedLMTrainer
from apex import amp

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
    parser.add_argument("--batch_sz", type=int, default=3, help="Batch size")
    parser.add_argument(
        "--aggregate_batch_sz",
        type=int,
        default=3,
        help="Batches are accumulated to this number before optimizer.step() is called. Must be a multiple of batch_sz.",
    )
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
        "--output_dir",
        type=str,
        default=".",
        help="Directory where checkpoints saves will be made.",
    )
    args = parser.parse_args()

    project_name = args.project_name
    epochs = args.epochs
    batch_size = args.batch_sz
    aggregate_batch_size = args.aggregate_batch_sz
    assert aggregate_batch_size % batch_size == 0
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
        "mem_len": 768,
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
        add_pads_to_target=True
    )
    val_set = ChunkedTextDataset(
        os.path.join(input_folder, "val.pt"),
        tokenizer,
        chunked_model_config["max_seq_len"],
        chunked_model_config["predict_len"],
        add_pads_to_target=True
    )
    train_loader = train_set.get_dataloader(batch_size, num_workers=1)
    val_loader = val_set.get_dataloader(batch_size, num_workers=0, random=False)

    # Initialize w&b logger
    do_wandb = False
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
        num_training_steps=epochs * len(train_set) / aggregate_batch_size,
    )

    # Shift model to device & enable fp16 if applicable.
    model.to(device)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    print("*** Running training ***")
    trainer = ChunkedLMTrainer(
        model,
        chunked_model_config,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        is_fp16=True,
        desired_batch_sz=aggregate_batch_size,
        do_wandb=do_wandb,
    )
    model.zero_grad()
    for _ in range(epochs):
        trainer.loop()

    trainer.loop(_validate=True, _skip_batches=10)
    trainer.save_model("final")

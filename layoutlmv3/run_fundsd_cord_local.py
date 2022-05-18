from examples.run_funsd_cord import main
import os


def local_main():
    args = []
    args.extend([
        "--model_name_or_path", "microsoft/layoutlmv3-base",
        "--dataset_name", "funsd",
        "--output_dir", "./output",
        "--do_train",
        "--do_eval",
        "--segment_level_layout", "1",
        "--visual_embed", "1",
        "--input_size", "224",
        "--max_steps", "1000",
        "--save_steps", "-1",
        "--evaluation_strategy", "steps",
        "--eval_steps", "100",
        "--per_device_train_batch_size", "4",
        "--overwrite_output_dir"
    ])
    main(args)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable GPU
    local_main()

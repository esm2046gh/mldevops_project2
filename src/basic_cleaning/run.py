#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()
    ###########################
    run = wandb.init(project="nyc_airbnb", group="eda", save_code=True)
    local_path = wandb.use_artifact(args.input_artifact).file()
    logger.info(f"Fetched latest version of artifact '{args.input_artifact}' from W&B. local_path: {local_path}")
    df = pd.read_csv(local_path)

    # 1) Dropping outliers
    logger.info(f"Dropping outliers from column 'price'. min_value: {args.min_price}, max_value: {args.max_price}")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # 2)  Convert last_review to datetime
    logger.info(f"Convert last_review to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    # 3)  Saving the cleaned artifact
    logger.info(f"Saving cleaned artifact as '{args.output_artifact}'")
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    df.to_csv(args.output_artifact, index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic data cleaning")
    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="The input artifact",
        required=True
    )
    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="The output artifact",
        required=True
    )
    parser.add_argument(
        "--output_type", 
        type=str,
        help="the output type",
        required=True
    )
    parser.add_argument(
        "--output_description", 
        type=str,
        help="The output description",
        required=True
    )
    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price",
        required=True
    )
    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price",
        required=True
    )
    args = parser.parse_args()
    go(args)

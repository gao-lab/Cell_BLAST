#!/usr/bin/env python

import sys
import os
import pathlib
import subprocess
import getpass
import numpy as np
import pandas as pd
import mysql.connector
from utils import nan_safe


def generate_datasets_meta():
    dataset_dict = {
        ds: (nb, [
            file for file in os.listdir(os.path.join(nb, ds))
            if file.endswith(".svg") and file != "peek.svg"
        ])
        for nb in snakemake.config["notebooks"]
        for ds in os.listdir(nb) if
            os.path.isdir(os.path.join(nb, ds)) and
            not ds.startswith(".") and
            ds != "__pycache__"
    }

    used_columns = (
        "dataset_name", "organism", "organ", "platform",
        "cell_number", "publication", "pmid", "remark"
    )
    single = pd.read_csv(
        "~/SC/Datasets/ACA_datasets.csv",  # TODO: don't use absolute path
        comment="#", skip_blank_lines=True
    ).loc[:, used_columns]
    additional = pd.read_csv(
        "~/SC/Datasets/additional_datasets.csv",  # TODO: don't use absolute path
        comment="#", skip_blank_lines=True
    ).loc[:, used_columns]
    single = pd.concat([single, additional], axis=0, ignore_index=True)
    aligned = pd.read_csv(
        "~/SC/Datasets/aligned_datasets.csv",  # TODO: don't use absolute path
        comment="#", skip_blank_lines=True
    ).loc[:, used_columns]

    for idx, row in aligned.iterrows():
        aligned.loc[idx, "cell_number"] = single.loc[np.in1d(
            single["dataset_name"], row["remark"].split(", ")
        ), "cell_number"].sum()

    combined = pd.concat([single, aligned], axis=0, ignore_index=True)
    # combined = combined.loc[np.in1d(
    #     combined["dataset_name"], list(dataset_dict.keys())
    # ), :]
    # combined["cell_number"] = combined["cell_number"].astype(np.int)

    combined["self-projection coverage"] = np.nan
    combined["self-projection accuracy"] = np.nan
    combined["predictable"] = np.nan
    combined["visualization"] = np.nan
    combined["notebook"] = np.nan
    combined["display"] = False
    for idx, row in combined.iterrows():
        if row["dataset_name"] not in dataset_dict:
            continue

        combined.loc[idx, "display"] = True
        combined.loc[idx, "notebook"] = dataset_dict[row["dataset_name"]][0]
        combined.loc[idx, "visualization"] = ", ".join(dataset_dict[row["dataset_name"]][1])
        spf_path = os.path.join(
            combined.loc[idx, "notebook"],
            row["dataset_name"],
            "self_projection.txt"
        )
        try:
            with open(spf_path, "r") as spf:
                lines = spf.readlines()
                k1, v1 = lines[0].split()
                k2, v2 = lines[1].split()
                assert k1 == "coverage" and k2 == "accuracy"
                v1, v2 = float(v1.strip()), float(v2.strip())
                combined.loc[idx, "self-projection coverage"] = v1
                combined.loc[idx, "self-projection accuracy"] = v2
        except Exception:
            print("Error reading self-projection metrics: " + spf_path)

        pf_path = os.path.join(
            combined.loc[idx, "notebook"],
            row["dataset_name"],
            "predictable.txt"
        )
        try:
            with open(pf_path, "r") as pf:
                combined.loc[idx, "predictable"] = ", ".join([
                    item.strip() for item in pf.readlines()
                ])
        except Exception:
            print("Error reading predictable variables: " + pf_path)

    return combined


def create_table(cursor):
    cursor.execute("DROP TABLE IF EXISTS `datasets`;")
    cursor.execute(
        "CREATE TABLE `datasets` ("
        "  `dataset_name` CHAR(50) NOT NULL UNIQUE,"
        "  `organism` char(50) NOT NULL,"
        "  `organ` char(100) NOT NULL,"
        "  `platform` char(50),"
        "  `cell_number` INT CHECK(`cell_number` > 0),"
        "  `publication` VARCHAR(300),"
        "  `pmid` CHAR(8),"
        "  `remark` VARCHAR(200),"
        "  `self-projection coverage` FLOAT CHECK(`self-projection coverage` BETWEEN 0 AND 1),"
        "  `self-projection accuracy` FLOAT CHECK(`self-projection accuracy` BETWEEN 0 AND 1),"
        "  `predictions` VARCHAR(200),"
        "  `visualization` VARCHAR(200),"
        "  `notebook` VARCHAR(100),"
        "  `display` BOOL NOT NULL,"
        "  PRIMARY KEY USING HASH(`dataset_name`)"
        ");"
    )


def insert_data(cursor, data):
    insert_sql = (
        "INSERT INTO `datasets` ("
        "  `dataset_name`, `organism`, `organ`, `platform`,"
        "  `cell_number`, `publication`, `pmid`, `remark`,"
        "  `self-projection coverage`, `self-projection accuracy`,"
        "  `predictions`, `visualization`, `notebook`, `display`"
        ") VALUES ("
        "  %s, %s, %s, %s,"
        "  %s, %s, %s, %s,"
        "  %s, %s,"
        "  %s, %s, %s, %s"
        ");"
    )
    cursor.executemany(insert_sql, [(
        nan_safe(row["dataset_name"]), nan_safe(row["organism"]),
        nan_safe(row["organ"]), nan_safe(row["platform"]),
        nan_safe(row["cell_number"], int), nan_safe(row["publication"]),
        nan_safe(row["pmid"], lambda x: str(int(x))), nan_safe(row["remark"]),
        nan_safe(row["self-projection coverage"], lambda x: float(np.round(x, 3))),
        nan_safe(row["self-projection accuracy"], lambda x: float(np.round(x, 3))),
        nan_safe(row["predictable"]), nan_safe(row["visualization"]),
        nan_safe(row["notebook"]), nan_safe(row["display"])
    ) for _, row in data.iterrows()])


def main():
    with open(snakemake.input.init, "r") as init:
        init_ret = subprocess.call("mysql -u caozj -p".split(" "), stdin=init)
    if init_ret:
        sys.exit(init_ret)
    while True:
        try:
            cnx = mysql.connector.connect(
                user=getpass.getuser(), password=getpass.getpass("Enter password: "),
                host="127.0.0.1", database="aca"
            )
        except mysql.connector.errors.ProgrammingError:
            print("Incorrect password! Try again.")
            continue
        except KeyboardInterrupt:
            print("Abort")
            sys.exit(1)
        break
    cursor = cnx.cursor()
    create_table(cursor)
    insert_data(cursor, generate_datasets_meta())
    cnx.commit()
    cursor.close()
    cnx.close()
    pathlib.Path(snakemake.output[0]).touch()


if __name__ == "__main__":
    main()

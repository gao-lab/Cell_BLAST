import os
import collections
import pathlib
import tempfile
import subprocess
import hashlib
import yaml
import numpy as np
import pandas as pd
import mysql.connector
from utils import nan_safe


representer = lambda self, data: yaml.representer.SafeRepresenter.represent_dict(self, data.items())
yaml.add_representer(dict, representer)
yaml.add_representer(collections.OrderedDict, representer)
COLUMNS = ["dataset_name", "version", "last", "curr", "remark"]


def create_table(cursor):
    cursor.execute("DROP TABLE IF EXISTS `history`, `versions`;")
    cursor.execute(
        "CREATE TABLE `versions` ("
        "  `version` CHAR(50) NOT NULL UNIQUE,"
        "  `time` DATETIME NOT NULL,"
        "  PRIMARY KEY USING HASH(`version`)"
        ");"
    )
    cursor.execute(
        "CREATE TABLE `history` ("
        "  `id` INT NOT NULL UNIQUE AUTO_INCREMENT,"
        "  `dataset_name` CHAR(50) NOT NULL,"
        "  `version` CHAR(50) NOT NULL,"
        "  `last` CHAR(32),"
        "  `curr` CHAR(32),"
        "  `remark` VARCHAR(500),"
        "  PRIMARY KEY(`id`),"
        "  FOREIGN KEY(`version`) REFERENCES `versions`(`version`)"
        ");"
    )


def insert_data(cursor, versions, history):
    insert_sql = (
        "INSERT INTO `versions` ("
        "  `version`, `time`"
        ") VALUES ("
        "  %s, %s"
        ");"
    )
    cursor.executemany(insert_sql, [
        (row["version"], row["time"])
        for _, row in versions.iterrows()
    ])
    insert_sql = (
        "INSERT INTO `history` ("
        "  `dataset_name`, `version`, `last`, `curr`, `remark`"
        ") VALUES ("
        "  %s, %s, %s, %s, %s"
        ");"
    )
    cursor.executemany(insert_sql, [
        (row["dataset_name"], row["version"], nan_safe(row["last"]), nan_safe(row["curr"]), row["remark"])
        for _, row in history.iterrows()
    ])


def main(snakemake):
    versions = pd.read_csv("release/versions.txt", sep="\t", names=["version", "time"])
    used_versions = {
        "curr": versions["version"].iloc[-1],
        "last": versions["version"].iloc[-2] if versions["version"].size > 1 else None
    }  # only need to look at the last two versions for changed items
    assert snakemake.wildcards.version == used_versions["curr"]

    md5sum = collections.OrderedDict()
    for key, val in used_versions.items():
        if val is None:
            md5sum[key] = pd.Series()
        else:
            md5sum[key] = pd.read_csv(
                f"release/md5sum-{val}.txt",
                sep=r"\s+", index_col=1, names=["md5sum"]
            )["md5sum"]
    md5sum = pd.DataFrame(md5sum, dtype=object)

    diff = md5sum.query("curr != last").copy()  # Query properly deals with nans
    diff.index.name = "dataset_name"
    diff.reset_index(inplace=True)
    diff["dataset_name"] = np.vectorize(lambda x: x.replace(".tar.gz", ""))(diff["dataset_name"])
    diff["version"] = used_versions["curr"]
    diff = diff.loc[:, COLUMNS[:-1]]

    try:  # In case downstream rules failed, we can reuse remarks if possible
        history = pd.read_csv("release/history.csv", dtype=object).loc[:, COLUMNS]
        existing = history.query(f"version == '{used_versions['curr']}'")
        history = history.query(f"version != '{used_versions['curr']}'")
    except FileNotFoundError:
        existing = history = pd.DataFrame(columns=COLUMNS, dtype=object)
    diff = diff.merge(existing, how="left")  # Only remarks with identical md5sum change are reused
    diff["remark"].fillna("Undocumented", inplace=True)

    # Interactively add remarks via text editor
    while diff.shape[0]:
        fd, remark_file = tempfile.mkstemp(suffix=".yml", text=True)
        with open(remark_file, "w") as f:
            yaml.dump(diff.set_index("dataset_name").to_dict(
                orient="index", into=collections.OrderedDict
            ), f)
        subprocess.run(["vim", remark_file]).check_returncode()
        with open(remark_file, "r") as f:
            diff = pd.DataFrame.from_dict(yaml.safe_load(f), orient="index", dtype=object)
            diff.index.name = "dataset_name"
            diff.reset_index(inplace=True)
            diff = diff.loc[:, COLUMNS]
        os.close(fd)
        undocumented = diff.query("remark == 'Undocumented'")
        if undocumented.shape[0]:
            print("The following undocumented changes are found:")
            print(undocumented)
            while True:
                ans = input("Leave these remarks as 'Undocumented'? (y/n) ")
                if ans not in ("y", "n"):
                    continue
                break
            if ans == "n":
                continue
        break

    history = pd.concat([history, diff], ignore_index=True)
    history.to_csv("release/history.csv", index=False)

    # Export to MySQL
    cnx = mysql.connector.connect(
        user=snakemake.config["db_user"], password=snakemake.config["db_passwd"],
        host="127.0.0.1", database="aca"
    )
    cursor = cnx.cursor()
    create_table(cursor)
    insert_data(cursor, versions, history)
    cnx.commit()
    cursor.close()
    cnx.close()
    pathlib.Path(snakemake.output[0]).touch()


if __name__ == "__main__":
    main(snakemake)

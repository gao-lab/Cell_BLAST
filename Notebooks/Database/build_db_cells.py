#!/usr/bin/env python

import sys
import pathlib
import getpass
import numpy as np
import pandas as pd
import mysql.connector
import Cell_BLAST as cb
from utils import nan_safe


def get_datasets(cnx):
    cursor = cnx.cursor(buffered=True)
    query = "SELECT `dataset_name`, `notebook` FROM `datasets` WHERE `display` = 1;"
    cursor.execute(query)
    for item in cursor:
        yield item
    cursor.close()


def get_data(dataset, notebook):
    obs = {
        key: val for key, val in cb.data.read_hybrid_path(
            "%s/%s/ref.h5//obs" % (notebook, dataset)
        ).items() if not (
            key.startswith("latent_") or
            key in ("organism", "organ", "platform", "__libsize__")
        )
    }
    obs["cid"] = cb.data.read_hybrid_path(
        "%s/%s/ref.h5//obs_names" % (notebook, dataset))
    obs = pd.DataFrame(obs)
    if "dataset_name" not in obs.columns.values:
        obs["dataset_name"] = dataset
    return obs


def native_type(x):
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def create_table(cnx, dataset, notebook):
    try:
        data = get_data(dataset, notebook)
    except OSError:  # TODO: remove this try-catch block after database is settled
        return
    cursor = cnx.cursor()
    drop_sql = "DROP TABLE IF EXISTS `%s`;" % dataset
    cursor.execute(drop_sql)
    create_sql = ["CREATE TABLE `%s` (" % dataset]
    for column in data.columns.values:
        if np.issubdtype(data[column].dtype.type, np.object_):
            if column in ("cid", "dataset_name"):
                dtype = "CHAR(50)"
            else:
                dtype = "CHAR(%d)" % np.vectorize(len)(data[column]).max()
        elif np.issubdtype(data[column].dtype.type, np.floating):
            dtype = "FLOAT"
        elif np.issubdtype(data[column].dtype.type, np.integer):
            dtype = "INT"
        else:
            raise Exception("Unexpected dtype!")
        if column == "cid":
            options = " UNIQUE NOT NULL"
        elif column == "dataset_name":
            options = " NOT NULL"
        else:
            options = ""
        create_sql.append("`%s` %s%s, " % (column, dtype, options))
    create_sql.append("PRIMARY KEY USING HASH(`cid`), ")
    create_sql.append("FOREIGN KEY(`dataset_name`) REFERENCES `datasets`(`dataset_name`));")
    create_sql = "".join(create_sql)
    cursor.execute(create_sql)

    columns = ", ".join(["`%s`" % item for item in data.columns.values])
    values = ", ".join(["%s"] * data.shape[1])
    insert_sql = "INSERT INTO `%s` (%s) VALUES (%s);" % (dataset, columns, values)
    value_pool = [tuple(
        nan_safe(row[column], native_type) for column in data.columns.values
    ) for _, row in data.iterrows()]
    while value_pool:  # Execute in chunks to avoid connection interruption
        chunk_size = min(len(value_pool), 5000)
        cursor.executemany(insert_sql, value_pool[:chunk_size])
        value_pool = value_pool[chunk_size:]
    cnx.commit()
    cursor.close()


def main():
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
    for dataset, notebook in get_datasets(cnx):
        print("Working on %s..." % dataset)
        create_table(cnx, dataset, notebook)
        cnx.commit()
    cnx.close()
    pathlib.Path(snakemake.output[0]).touch()


if __name__ == "__main__":
    main()

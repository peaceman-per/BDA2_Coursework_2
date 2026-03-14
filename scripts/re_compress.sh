#!/usr/bin/env bash

# turns out, some of the parquet files are compressed in zstd
# this cluster (lovely lena) does not support the decompression of zstd
# we must re-encode them on the local machine

# run on the local cluster machine with PyArrow library
set -euo pipefail

HDFS_BASE="/user/wsidn001/bda2/coursework-2/dat"
LOCAL_TMP="/tmp/parquet_recompress"
mkdir -p "$LOCAL_TMP"

# first, identify which files are zstd-compressed
# for this, it was years 2022, 2023, 2024, 2025

SERVICES="yellow green fhv"

for service in $SERVICES; do
    for year in 2024 2025; do
        for month in $(seq -w 1 12); do
            FILE="${service}_tripdata_${year}-${month}.parquet"
            HDFS_PATH="${HDFS_BASE}/${year}/${FILE}"

            # check if file exists on HDFS
            if ! hdfs dfs -test -e "$HDFS_PATH" 2>/dev/null; then
                echo "SKIP (not found): $FILE"
                continue
            fi

            echo "Processing: $FILE"

            # 1 pull from hdfs to local disk
            hdfs dfs -get "$HDFS_PATH" "$LOCAL_TMP/$FILE"

            # 2 re-compress with snappy
            python3 -c "
import pyarrow.parquet as pq
reader = pq.ParquetFile('$LOCAL_TMP/$FILE')
table = reader.read()
pq.write_table(table, '$LOCAL_TMP/${FILE}.snappy', compression='snappy')
print('  Converted: $FILE')
"

            # 3 push back to hdfs (overwrite original)
            hdfs dfs -put -f "$LOCAL_TMP/${FILE}.snappy" "$HDFS_PATH"
            echo "  Uploaded:  $FILE"

            # 4 clean up local disk
            rm -f "$LOCAL_TMP/$FILE" "$LOCAL_TMP/${FILE}.snappy"

        done
    done
done

echo "Done! All zstd files re-compressed to snappy."
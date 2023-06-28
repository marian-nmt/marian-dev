# marian-YYYY-MM-DD-revision.tgz
# This combines marian, marian_decoder in a single TAR file for
# execution in MSFT internal tools FLO and Singularity.

execute_process(
        COMMAND bash -c "TZ=America/Los_Angeles date +%Y-%m-%d"
        OUTPUT_VARIABLE TGZ_DATE
        OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
        COMMAND git rev-parse --short=7 HEAD
        OUTPUT_VARIABLE TGZ_REV
        OUTPUT_STRIP_TRAILING_WHITESPACE)

message("Generating ${CWD}/marian-${TGZ_DATE}-${TGZ_REV}.tgz")

# check if pigz is available for faster compression
execute_process(
        COMMAND bash -c "which pigz || which gzip"
        OUTPUT_VARIABLE COMPRESS
        OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
        COMMAND tar -I ${COMPRESS} -cvvf "${CWD}/marian-${TGZ_DATE}-${TGZ_REV}.tgz" -C "${CWD}"
            marian 
            marian-decoder 
            marian-scorer 
            marian-vocab 
            marian-conv
        WORKING_DIRECTORY "${CWD}")        
# README
## Description
Program to investigate the response of FREDDA to burst morphology and RFI. Coherently redisperses HTR data from CELEBI to a range of trial DMs and reinjects them into FREDDA to determine

## Additional dependencies
CRAFT repository for FREDDA executable (https://github.com/askap-craco/craft). It is also necessary to have old executables of FREDDA for older FRBs (prior to 200406). The executables should be placed in a folder called `fredda` and older versions should be named `cudafdmt_190101` and `cudafdmt_200228`. This folder with the relevant executables is available on ozstar at `/fred/oz002/jhoffmann/RedisperseFRB/fredda/`.

## Running the pipeline
To run the entire pipeline for a single FRB run main.nf in the pipelines folder.

    nextflow run main.nf --frb=<FRB_NAME> --DMi=<DM_FROM_CELEBI> --t_res=<TIME_RESOLUTION> --f_mid=<CENTRAL_OBS_FQ> --max_dm=<MAX_REDISPERSION_DM>

If additional filterbanks are required after it has already been run once, add the flag `--first_run=false` to save computational resources. The other parameters are identical.

To obtain the maximum searched DM of FREDDA run:

    ./scripts/get_max_dm.sh <CENTRAL_OBS_FQ> <TIME_RESOLUTION>

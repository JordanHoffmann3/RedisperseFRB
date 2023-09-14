# README
## Description
Program to investigate the response of FREDDA to burst morphology and RFI. Coherently redisperses HTR data from CELEBI to a range of trial DMs and reinjects them into FREDDA to determine

## Additional dependencies and setup
CRAFT repository for FREDDA executable (https://github.com/askap-craco/craft). It is also necessary to have old executables of FREDDA for older FRBs (prior to 200406). The executables should be placed in a folder called `fredda` and older versions should be named `cudafdmt_190101` and `cudafdmt_200228`. This folder with the relevant executables is available on ozstar at `/fred/oz002/jhoffmann/RedisperseFRB/fredda/`.

The pipeline also assumes a `$REDIS` bash variable which points to this `RedisperseFRB` folder. The output files in `pipeline/main.nf` will need to be specified in the command line or edited in the file.

## Running the pipeline
To run the entire pipeline for a single FRB run main.nf in the pipeline folder.

    nextflow run main.nf --frb=<FRB_NAME> --DMi=<DM_FROM_CELEBI> --t_res=<TIME_RESOLUTION> --f_mid=<CENTRAL_OBS_FQ> --max_dm=<MAX_REDISPERSION_DM>

If additional filterbanks are required after it has already been run once, add the flag `--first_run=false` to save computational resources. The other parameters are identical. This allows the pipeline to use results from previous intermediate results used to produce the filterbanks. If `--first_run=true` is run twice, it will create a completely new folder and rename the old one.

## Outputs
Outputs are written to the folder `params.out_dir` in pipeline/main.nf which defaults to `output/Dispersed_FRB`. The final SNR curves used in the zdm code are written to `params.SNR_dir` in pipeline/main.nf which defaults to `output/SNR_curves`. PDF plots of the curves as well as the numpy files are saved.

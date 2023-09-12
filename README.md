\title{README}

\hline

\title{Purpose}
Program to investigate the response of FREDDA to burst morphology and RFI.

\hline
\title{Dependencies}
CRAFT repository for FREDDA executable (https://github.com/askap-craco/craft)

\hline
\title{Running}
To run the entire pipeline for a single FRB run main.nf in the pipelines folder.

nextflow run main.nf --frb=<FRB_NAME> --DMi=<DM_FROM_CELEBI> --t_res=<TIME_RESOLUTION> --f_mid=<CENTRAL_OBS_FQ> --max_dm=<MAX_REDISPERSION_DM>

To obtain the maximum searched DM of FREDDA run ./scripts/get_max_dm.sh <CENTRAL_OBS_FQ> <TIME_RESOLUTION>

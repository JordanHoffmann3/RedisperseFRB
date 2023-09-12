nextflow.enable.dsl=2

params.frb = "000000"       // FRB name
params.DMi = 0.0            // Observed DM
params.f_mid = 1271.5       // Central frequency in MHz
params.t_res = 1.182        // Time res in ms

params.first_run = true

params.max_dm = 100
params.dm_res = 50 
params.data_dir = "/fred/oz002/askap/craft/craco/processing/output"
params.out_dir = "/fred/oz002/jhoffmann/RedisperseFRB/Outputs/Dispersed_${params.frb}"
params.publish_dir = "${params.out_dir}/outputs"
params.f = "${params.out_dir}/data/${params.frb}_param.dat" 
params.pos = -1

/*
 *  Make directory structure
 */
process mk_dirs {

    input:

    output:
        val true
    
    script:
        """
        if [ -d ${params.out_dir} ]
        then
            i=1
            while [ -d ${params.out_dir}_\$i ]
            do
                i=\$(bc -l <<< "\$i + 1")
            done

            mv ${params.out_dir} ${params.out_dir}_\$i
        fi

        mkdir ${params.out_dir}/
        mkdir ${params.out_dir}/data
        mkdir ${params.publish_dir}
        mkdir ${params.out_dir}/fredda_outputs
        echo "# DM, S/N, sampno, secs from file start, boxcar, idt, dm, beamno,mjd, sampno_start, sampno_end, idt_start, idt_end, ncands" > ${params.out_dir}/fredda_outputs/extracted_outputs.txt

        echo ${params.frb} > ${params.f}
        echo ${params.DMi} >> ${params.f}
        echo ${params.f_mid} >> ${params.f}
        echo ${params.t_res} >> ${params.f}
        """
}

/*
 *  Get HTR data and extract 1s of data
 */
process extract_frb {
    executor 'slurm'
    memory '32 GB'
    time '5m'

    input:
        val ready

    output:
        val true

    script:
        """
        x_t=${params.data_dir}/${params.frb}/htr/${params.frb}_X_t_${params.DMi}.npy
        y_t=${params.data_dir}/${params.frb}/htr/${params.frb}_Y_t_${params.DMi}.npy

        cd \$REDIS
        python src/plot_HTR.py ${params.frb} --dt=1 --buf=1000 -p ${params.pos} -o "${params.out_dir}/data/" -f \$x_t \$y_t ${params.f}
        """
}

/*
 *  First job
 */
process job_first {
    executor 'slurm'
    memory '80 GB'
    time '30m'
 
    input:
        val ready

    output:
        val true

    script:
        """
        cd \$REDIS

        # Remove existing DM 0 file in case it was produced with different flags
        if [ -f ${params.publish_dir}/${params.frb}_DM_0.0.npy ]
        then
            rm ${params.publish_dir}/${params.frb}_DM_0.0.npy
        fi

        python src/redisperse.py ${params.frb} ${params.DMi} 0.0 ${params.f_mid} ${params.t_res} --out_dir=${params.publish_dir} --data_dir=${params.out_dir}/data/ -s
        """
}

/*
 *  Job for specific DM     
 */
process job_single {
    executor 'slurm'
    memory '40 GB'
    time '10m'

    input:
        val ready
        val DM_in

    output:
        val DM_in
    
    // when:
    //     file("${params.publish_dir}/${params.frb}_DM_${DM_in}.fil").exists() == false

    script:
        if (file("${params.publish_dir}/${params.frb}_DM_${DM_in}.fil").exists() == false)
             """
            python \$REDIS/src/redisperse.py ${params.frb} ${params.DMi} ${DM_in} ${params.f_mid} ${params.t_res} --out_dir=${params.publish_dir} --data_dir=${params.out_dir}/data/
            """
        else
            """
            echo "${params.publish_dir}/${params.frb}_DM_${DM_in}.fil already exists - skipping redispersion"
            """
}
 
/*
 *  FREDDA run for specific DM     
 */
process fredda_single {
    if (file("${params.out_dir}/fredda_outputs/${params.frb}_DM_${DM}.fil.cand.fof").exists() == false)
        executor 'slurm'
        // accelerator 1, type: 'nvidia-tesla-p100'
        clusterOptions '--gres=gpu'
        memory '8 GB'
        time '2m'
        errorStrategy 'ignore'
        // errorStrategy { sleep(1); task.exitStatus == 134 ? 'retry' : 'ignore' }
        // maxRetries 3

    input:
        val DM

    output:
        val true
        // val DM
        // path "${params.out_dir}/fredda_outputs/*.cand.fof"

    script:
        if (file("${params.out_dir}/fredda_outputs/${params.frb}_DM_${DM}.fil.cand.fof").exists() == false)
            """
            cd ${params.out_dir}
            source \$REDIS/loadcuda.sh > /dev/null

            f="${params.frb}_DM_${DM}.fil"

            \$runcudafdmt ${params.publish_dir}/\${f} -t 1024 -d 4096 -x 9 -o fredda_outputs/\${f}.cand
            python \$REDIS/fredda/fredfof.py --dmmin=-1.0 fredda_outputs/\${f}.cand

            # printf "%s \t" ${DM} >> ${params.out_dir}/fredda_outputs/extracted_outputs.txt
            # echo \$(sort -k 1 -n ${params.out_dir}/fredda_outputs/\${f}.cand.fof | tail -1) >> ${params.out_dir}/fredda_outputs/extracted_outputs.txt
            """
        else
            """
            echo "${params.out_dir}/fredda_outputs/${params.frb}_DM_${DM}.fil.cand.fof already exists - skipping fredda run"
            """
}

/*
 *  Reverse frequencies for old versions to run
 */
process reverse_fq {
    input:
        val DM

    output:
        val DM

    script:
        if (file("${params.out_dir}/reverse_fq/${params.frb}_DM_${DM}.fil").exists() == false)
            """
            if [ ! -d ${params.out_dir}/reverse_fq ]
            then
                mkdir ${params.out_dir}/reverse_fq
            fi

            f=${params.frb}_DM_${DM}.fil

            cd ${params.publish_dir}
            foff=\$(python \$REDIS/src/flip_frequencies.py \$f ${params.out_dir}/reverse_fq/\$f)
            if echo "\$foff < 0.0" | bc -l | grep -q 1
            then
                mv \$f temp_${DM}.fil
                mv ${params.out_dir}/reverse_fq/\$f \$f
                mv temp_${DM}.fil ${params.out_dir}/reverse_fq/\$f
            fi
            """
        else
            """
            """
}

/*
 *  Run old version
 */
process fredda_reverted {
    executor 'slurm'
    // accelerator 1, type: 'nvidia-tesla-p100'
    clusterOptions '--gres=gpu'
    memory '8 GB'
    time '2m'
    errorStrategy 'ignore'
    // errorStrategy { sleep(1); task.exitStatus == 134 ? 'retry' : 'ignore' }
    // maxRetries 3

    input:
        val DM
    
    output:
        val true
        
    script:
        if (file("${params.out_dir}/fredda_reverted/${params.frb}_DM_${DM}.fil.cand.fof").exists() == false)
            """
            if [ ! -d ${params.out_dir}/fredda_reverted ]
            then
                mkdir ${params.out_dir}/fredda_reverted
                echo "# DM, S/N, sampno, secs from file start, boxcar, idt, dm, beamno,mjd, sampno_start, sampno_end, idt_start, idt_end, ncands" > ${params.out_dir}/fredda_reverted/extracted_outputs.txt
            fi

            f=${params.frb}_DM_${DM}.fil

            if [ ${params.frb} -lt 190830 ]
            then
                date=190101
            else
                date=200228
            fi
            
            source \$REDIS/loadcuda.sh > /dev/null
            \${runcudafdmt}_\${date} ${params.out_dir}/reverse_fq/\$f -t 1024 -d 4096 -x 9 -o ${params.out_dir}/fredda_reverted/\${f}.cand
            
            if [ \$(awk '{print NF}' ${params.out_dir}/fredda_reverted/\${f}.cand | tail -n 1) == '7' ]
            then
        
                while read -r line
                do
                    if [ "\${line:0:1}" == "#" ]
                    then
                        echo "\$line, mjd" > temp.txt
                    else
                        echo "\$line 0" >> temp.txt
                    fi
                done < "${params.out_dir}/fredda_reverted/\${f}.cand"
                mv temp.txt ${params.out_dir}/fredda_reverted/\${f}.cand
                
            fi

            source \$REDIS/loadpy.sh > /dev/null
            python \$REDIS/fredda/fredfof.py ${params.out_dir}/fredda_reverted/\${f}.cand

            # printf "%s \t" ${DM} >> ${params.out_dir}/fredda_reverted/extracted_outputs.txt
            # echo \$(sort -k 1 -n ${params.out_dir}/fredda_reverted/\${f}.cand.fof | tail -1) >> ${params.out_dir}/fredda_reverted/extracted_outputs.txt
            """
        else
            """
            echo "${params.out_dir}/fredda_reverted/\${f}.cand.fof already exists - skipping old fredda run"
            """
}

/* 
 *  Extract outputs
 */
process extract {
    input:
        val go

    script:
        """
        cd ${params.out_dir}/fredda_outputs/
        echo "# DM, S/N, sampno, secs from file start, boxcar, idt, dm, beamno,mjd, sampno_start, sampno_end, idt_start, idt_end, ncands" > extracted_outputs.txt

        for f in *.cand.fof
        do
            DM="\$(echo \$f | awk -F'[_.]' '{print \$3}').\$(echo \$f | awk -F'[_.]' '{print \$4}')"
            printf "%s \t" \${DM} >> temp.txt
            line=\$(sort -k 1 -n \$f | tail -1)
            echo \${line} >> temp.txt
        done

        sort -k 1 -n temp.txt >> extracted_outputs.txt
        rm temp.txt
        """
}

/* 
 *  Extract reverted outputs
 */
process extract_reverted {
    input:
        val go

    script:
        """
        cd ${params.out_dir}/fredda_reverted/
        echo "# DM, S/N, sampno, secs from file start, boxcar, idt, dm, beamno,mjd, sampno_start, sampno_end, idt_start, idt_end, ncands" > extracted_outputs.txt

        for f in *.cand.fof
        do
            DM="\$(echo \$f | awk -F'[_.]' '{print \$3}').\$(echo \$f | awk -F'[_.]' '{print \$4}')"
            printf "%s \t" \${DM} >> temp.txt
            line=\$(sort -k 1 -n \$f | tail -1)
            echo \${line} >> temp.txt
        done

        sort -k 1 -n temp.txt >> extracted_outputs.txt
        rm temp.txt
        """
}

/*
 * Define the workflow
 */
workflow {
    dms = Channel.from(0..params.max_dm).filter{it%params.dm_res==0}.toDouble()

    if (params.first_run == true) {
        println "First run"
        mk_dirs()
        extract_frb(mk_dirs.out)
        job_first(extract_frb.out)
        job_single(job_first.out, dms)
        fredda_single(job_single.out)
        extract(fredda_single.out.collect())

        if (((String)params.frb)[0..5] as int < 200406) {
            reverse_fq(job_single.out)
            fredda_reverted(reverse_fq.out)
            extract_reverted(fredda_reverted.out.collect())
        }
    }
    else {
        println "Second run"
        job_single(true, dms)
        fredda_single(job_single.out)
        extract(fredda_single.out.collect())

        if (((String)params.frb)[0..5] as int  < 200406) {
            reverse_fq(job_single.out)
            fredda_reverted(reverse_fq.out)
            extract_reverted(fredda_reverted.out.collect())
        }
    }
}
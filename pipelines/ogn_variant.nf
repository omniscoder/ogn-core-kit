#!/usr/bin/env nextflow

/*
 * OGN Germline Variant Pipeline (Nextflow)
 *
 * Stages:
 *   1. FASTQ/CRAM ingest + QC
 *   2. Alignment with BWA-MEM2 (CPU)
 *   3. GPU variant calling via `ogn_variant_runner`
 *
 * Usage:
 *   nextflow run pipelines/ogn_variant.nf \
 *     --sample_id HG002 \
 *     --reads "data/fastq/*.fastq.gz" \
 *     --reference data/ref/GRCh38.fa \
 *     --engine_image <engine_container_image> \
 *     --output out
 *
 * GPU knobs surface to Nextflow via params:
 *   --streams_per_gpu 4
 *   --max_active_regions 128
 *   --dl_model path/to/deepvariant.trt
 */

nextflow.enable.dsl=2

params.sample_id        = params.sample_id ?: "ogn-sample"
params.reads            = params.reads ?: ""
params.cram             = params.cram ?: ""
params.reference        = params.reference ?: ""
params.engine_image     = params.engine_image ?: null
params.output           = params.output ?: "ogn_results"
params.threads          = params.threads ?: 32
params.bwa_index        = params.bwa_index ?: ""
params.streams_per_gpu  = params.streams_per_gpu ?: 1
params.max_active       = params.max_active ?: 64
params.dl_model         = params.dl_model ?: ""
params.dl_hint          = params.dl_hint ?: ""
params.dl_streams       = params.dl_streams ?: 2
params.dl_batch         = params.dl_batch ?: 32
params.dl_cpu           = params.dl_cpu ?: false

workflow {
    take:
        reads_ch = params.reads ? Channel.fromPath(params.reads) : Channel.empty()
        cram_ch  = params.cram  ? Channel.value(params.cram)     : Channel.empty()

    main:
        qc_out = qc_stage(reads_ch)

        bam_out = map_stage(
            qc_out,
            params.reference,
            params.bwa_index,
            params.sample_id,
            params.threads
        )

        call_out = gpu_variant_stage(
            bam_out,
            params.reference,
            params.sample_id,
            params.streams_per_gpu,
            params.max_active,
            params.dl_model,
            params.dl_hint,
            params.dl_streams,
            params.dl_batch,
            params.dl_cpu
        )

    emit:
        call_out.view { "VCF: ${it}" }
}

process qc_stage {
    tag "${sample_id}"
    publishDir "${params.output}/qc", mode: 'copy'
    errorStrategy 'terminate'

    input:
        path read_file

    output:
        tuple val(sample_id), path("${sample_id}_R1.fastq.gz")
        tuple val(sample_id), path("${sample_id}_R2.fastq.gz")

    script:
        sample_id = read_file.baseName.replaceAll(/(_R[12])?(\.fastq(\.gz)?)$/, "")
        """
        mkdir -p ${sample_id}
        # Placeholder QC (fastp recommended)
        fastp -i ${read_file} -I ${read_file} \\
              -o ${sample_id}_R1.fastq.gz \\
              -O ${sample_id}_R2.fastq.gz \\
              -h ${sample_id}/${sample_id}.html \\
              -j ${sample_id}/${sample_id}.json
        """
}

process map_stage {
    tag "${sample_id}"
    publishDir "${params.output}/bam", mode: 'copy'
    errorStrategy 'terminate'

    input:
        tuple val(sample_id), path(read1), path(read2)

    output:
        tuple val(sample_id), path("${sample_id}.sorted.bam")

    script:
        reference = params.reference
        bwa_index = params.bwa_index ?: reference
        """
        if [ ! -f "${reference}.bwt.2bit.64" ]; then
            bwa-mem2 index ${reference}
        fi

        bwa-mem2 mem -t ${params.threads} ${bwa_index} ${read1} ${read2} |
            samtools sort -@ ${params.threads} -o ${sample_id}.sorted.bam

        samtools index ${sample_id}.sorted.bam
        """
}

process gpu_variant_stage {
    tag "${sample_id}"
    publishDir "${params.output}/vcf", mode: 'copy'
    errorStrategy 'terminate'
    container params.engine_image

    input:
        tuple val(sample_id), path(sorted_bam)

    output:
        path("${sample_id}.vcf.gz")

    script:
        reference = params.reference
        dl_model  = params.dl_model ? "--dl-model ${params.dl_model}" : ""
        dl_hint   = params.dl_hint  ? "--dl-hint ${params.dl_hint}"   : ""
        dl_cpu    = params.dl_cpu   ? "--dl-cpu"                      : ""

        """
        ogn_variant_runner \\
          --sample ${sample_id} \\
          --cram ${sorted_bam} \\
          --reference ${reference} \\
          --streams-per-gpu ${params.streams_per_gpu} \\
          --max-active-regions ${params.max_active} \\
          --dl-streams ${params.dl_streams} \\
          --dl-batch ${params.dl_batch} \\
          ${dl_model} ${dl_hint} ${dl_cpu} \\
          | bgzip -c > ${sample_id}.vcf.gz
        tabix -p vcf ${sample_id}.vcf.gz
        """
}

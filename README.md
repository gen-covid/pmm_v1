# Post-Mendelian genetic model in COVID-19

* boolean_features.py: Generation of the boolean representation from variants annoted with VEP

* analysis.py: Feature extraction by logistic regression with LASSO regularization

For computing the adjusted phenotype, a CSV file with the following format is required:
* One column named sample with the unique ID of the sample
* One column named age with age in years
* One column named gender with the sex coded as 0/1, with 1 for females
* One column named grading coded as:
  * 0 = not hospitalized
  * 1 = hospitalized without respiratory support
  * 2 = hospitalized with O2 supplementation
  * 3 = hospitalized with CPAP-biPAP
  * 4  = hospitalized intubated
Only PCR-positive individuals should be included.

## Boolean representations
For all the analyses only variants that are classified with impact HIGH or MODERATE in Ensemble are retained, namely: 'transcript_ablation', 'splice_acceptor_variant', 'splice_donor_variant', 'stop_gained', 'frameshift_variant', 'stop_lost', 'start_lost', 'transcript_amplification', 'inframe_insertion', 'inframe_deletion', 'missense_variant', 'protein_altering_variant'.

The required inputs for this step are the VCF files and VEP annotation files divided by chromosomes and sorted by ensembl gene code. GRCh38 is assumed as a reference genome.

The boolean representations are described in the following subsections. Each boolean representation is saved in a CSV file with corresponding name, which is then used for fitting the model.

### data_al1_rare
The following variants are selected:
Variants with frequency in gnomAD_NFE/ExAC_EUR < 1%
Variants that are classified as Pathogenic or Likely Pathogenic in CLINSIG with frequency in gnomAD_NFE/ExAC_EUR < 5%
Feature i,j is 1 if gene i in sample j has at least one variant in the previous list (it doesn’t matter if the variant is in homozygosis or heterozygosis).

### data_al2_rare
The following variants are selected:
Variants with frequency in gnomAD_NFE/ExAC_EUR < 1%
Variants that are classified as Pathogenic or Likely Pathogenic in CLINSIG with frequency in gnomAD_NFE/ExAC_EUR < 5%
Feature i,j is 1 if gene i in sample j has at least one homozygous variant or two heterozygous variants in the previous list. For male subjects, hemizygous variants on chromosome X are also considered.

### data_gc_hetero
Variants with frequency in gnomAD_NFE/ExAC_EUR > 5% are selected. For each gene, all the possible combinations of variants in the dataset are calculated, without considering if the variants are heterozygous or homozygous. Only the combinations with frequency > 5% are retained. Feature i,j is 1 if the combination of common variants i is present in sample j. Which variants define each combination is reported in the corresponding txt file.

### data_gc_homo
Variants with frequency in gnomAD_NFE/ExAC_EUR > 5% are selected. For each gene, all the possible combinations of homozygous variants in the dataset are calculated. Only the combinations with frequency > 5% are retained. Feature i,j is 1 if the combination of common variants i is present in sample j. Which variants define each combination is reported in the corresponding txt file.

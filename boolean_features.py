import os
import glob
import pandas as pd
import numpy as np

home_vcf_annot = '.' # CHANGE THIS TO THE FOLDER CONTAINING YOUR VCF AND ANNOTATION FILES
vcf_prefix = 'joint.5' # CHANGE THIS TO THE PREFIX OF YOUR VCF FILES, in my case files are named: ../joint.5.chr*.vcf.gz
annot_prefix = 'finalAnnot.SortByGene' # CHANGE THIS TO THE PREFIX OF YOUR VEP ANNOTATION FILES, in my case the files are named: ../finalAnnot.SortByGene.chr*.txt
pheno_name = 'phenotype.csv'
frequency_rare = 0.01
frequency_gc = 0.05
frequency_common = 0.05
ens2name = pd.read_csv('ensemblid_names.csv')

def update_variants_per_gene(data, bool_single, bool_multi, samples, gene):
    gene_name = np.unique(ens2name['Gene name'].loc[ens2name['Gene stable ID'] == gene].values)[0]
    gene_with = np.any(data, axis = 0).astype(int)
    bool_single[gene_name] = {sample:gt for sample, gt in zip(samples, gene_with)}
    gene_with_multi = (np.sum(data, axis = 0) > 1).astype(int)
    bool_multi[gene_name] = {sample:gt for sample, gt in zip(samples, gene_with_multi)}

def update_gc_per_gene(data, infos,  hetero_homo, bool_aplo, fout, samples, gene):
    gene_name = np.unique(ens2name['Gene name'].loc[ens2name['Gene stable ID'] == gene].values)[0]
    n_rows_gene = 0
    n_samples_poli_rare = 0
    if hetero_homo == 'hetero':
        gts_bool = data.astype(bool)
    elif hetero_homo == 'homo':
        gts_bool = (data > 1).astype(bool)
    else:
        raise ValueError('ERROR: wrong value for hetero_homo parameter')
    is_mut = np.any(gts_bool, axis = 0).astype(int)
    poli, poli_inv, counts = np.unique(gts_bool, axis = 1, return_inverse = True, return_counts= True)
    freq = counts / np.sum(counts)
    bool_aplo['{}_{}'.format(gene_name,0)] =  {sample:0 for sample in samples} 
    for i_poli in range(poli.shape[1]):
        if not np.any(poli[:,i_poli]):
            continue # no variants at all
        if freq[i_poli] < frequency_gc:
            for i_sample, sample in enumerate(samples):
                if poli_inv[i_sample] == i_poli:
                    bool_aplo['{}_{}'.format(gene_name,0)][sample] = 1
                    n_samples_poli_rare += 1
        else:
            n_rows_gene += 1
            row_name = ''
            for index, variant in infos[poli[:,i_poli]].iterrows():
                row_name += '{}:{}-{}:{}-{}_,'.format(variant['chrom'], variant['start'], variant['end'], variant['ref'], variant['alt'])
            bool_aplo['{}_{}'.format(gene_name, n_rows_gene)] = {sample:0 for sample in samples}
            n_samples_this_poli = 0
            for i_sample, sample in enumerate(samples):
                if poli_inv[i_sample] == i_poli:
                    bool_aplo['{}_{}'.format(gene_name, n_rows_gene)][sample] = 1
                    n_samples_this_poli += 1
            fout.write('{}_{} = '.format(gene_name, n_rows_gene))
            fout.write('{} = {}\n'.format(row_name, n_samples_this_poli))
    if n_samples_poli_rare > 0:
        fout.write('{}_{} = '.format(gene_name, 0))
        fout.write('{} = {}\n'.format('rare', n_samples_poli_rare))
    else:
        del bool_aplo['{}_{}'.format(gene_name,0)]
    fout.flush()

#--- Initialization
bool_single_rare = {}
bool_multi_rare = {}
bool_gc_homo = {}
bool_gc_hetero = {}
fout_gc_homo = open('bool_gc_homo.txt','wt')
fout_gc_hetero = open('bool_gc_hetero.txt','wt')
contigs = ['chr{}'.format(ind) for ind in range(1,23)] + ['chrX', 'chrY']
n_genes = 0
pheno = pd.read_csv(pheno_name, index_col = 0)

for contig in contigs:
    vcf = '{}/{}.{}.vcf.gz'.format(home_vcf_annot, vcf_prefix, contig)
    if not os.path.exists(vcf):
        raise ValueError('ERROR: missing file {}'.format(vcf))
    if not os.path.exists(vcf+'.tbi'):
        raise ValueError('ERROR: missing file {}'.format(vcf+'.tbi'))
    ann = '{}/{}.{}.txt'.format(home_vcf_annot, annot_prefix, contig)
    if not os.path.exists(ann):
        raise ValueError('ERROR: missing file {}'.format(ann))
    cmd = 'bcftools view -h ./{} > header.txt'.format(vcf)
    os.system(cmd)
    samples = None
    with open('./header.txt','rt') as fin:
        for l in fin.readlines():
            l = l.strip().split()
            if l[0] == '#CHROM':
                samples = l[9:]
    if samples is None:
        raise ValueError('ERROR: wrong format in {}'.format(vcf))
    genders = []
    for sample in samples:
        try:
            sample_sex = pheno.loc[sample]['gender']
        except:
            sample_sex = 1
        genders.append(sample_sex)
    print('Number of samples for chrom {}: {}'.format(contig, len(samples)))
    fin_ann = open(ann)
    l = fin_ann.readline()
    gene_old = None
    gene_data = []
    while l:
        l = l.strip()
        if l:
            if l[0] != '#':
                l = l.split()
                idd = l[0]
                chrom = l[0].split(':')[0]
                ref = l[0].split(':')[2]
                alt = l[0].split(':')[3]
                pos = l[1].split(':')[1]
                start = int(pos.split('-')[0])
                if '-' in pos:
                    end = int(pos.split('-')[1])
                else:
                    end = start
                gene = l[3]
                feat_type = l[5]
                cons = l[6]
                extra = l[13]
                if 'gnomAD_NFE_AF' in extra:
                    ind = extra.index('gnomAD_NFE_AF=')
                    af = float(extra[ind:].split(';')[0].split('=')[1])
                else:
                    af = 1.0
                if 'CLIN_SIG' in extra:
                    ind = extra.index('CLIN_SIG=')
                    clin_sig = extra[ind:].split(';')[0].split('=')[1]
                else:
                    clin_sig = ''
                if (gene != gene_old):
                    if (gene_old is not None) and ('ENSG' in gene_old):
                        info = pd.DataFrame(gene_data, columns = ['id', 'chrom', 'start', 'end', 'ref', 'alt', 'gene', 'feat_type', 'cons', 'af', 'clin_sig'])
                        info.drop_duplicates(subset = 'id', inplace = True)
                        info.sort_values(by = 'id', axis = 0, inplace = True)
                        with open('ids.txt','wt') as fout:
                            for i in info['id']:
                                fout.write(i+'\n')
                        cmd = 'bcftools filter -r {}:{}-{} -Ou ./{} | bcftools view -i "%ID=@./ids.txt" -Ou | bcftools query -f "%ID [ %GT] \n" > ./variants.txt'.format(chrom, min(info['start']), max(info['start']), vcf)
                        os.system(cmd)
                        gts = []
                        with open('variants.txt','rt') as fin:
                            for l in fin.readlines():
                                l = l.strip()
                                if l:
                                    l = l.split()
                                    row = [l[0],]
                                    for gt, gender in zip(l[1:], genders):
                                        if ((gt == '1/1') or (gt == '1|1')) or (
                                            ((gt == '0/1') or (gt == '0|1') or (gt == '1/0') or (gt == '1|0')) and (gender == 1) ):
                                            row.append(2)
                                        elif (gt == '0/1') or (gt == '0|1') or (gt == '1/0') or (gt == '1|0'):
                                            row.append(1)
                                        elif (gt == '0/0') or (gt == '0|0') or (gt == './.') or (gt == '.|.'):
                                            row.append(0)
                                        else:
                                            raise ValueError('ERROR: unexpected genotype {}'.format(gt))
                                    gts.append(row)
                        gts = pd.DataFrame(gts, columns = ['id',]+samples)
                        gts.sort_values(by = 'id', axis = 0, inplace = True)
                        data = info.merge(gts, left_on = 'id', right_on = 'id', how = 'inner')
                        keep = np.logical_or.reduce( [data['cons'].str.contains(variant_type) for variant_type in ['transcript_ablation', 'splice_acceptor_variant', 'splice_donor_variant', 'stop_gained', 'frameshift_variant', 'stop_lost', 'start_lost', 'transcript_amplification', 'inframe_insertion', 'inframe_deletion', 'missense_variant', 'protein_altering_variant']] )
                        pat = data['clin_sig'].str.contains('athogenic')
                        var1 = np.logical_and.reduce( ( keep, np.logical_or(data['af'] < frequency_rare, data['af'] == 1.0) ) )
                        var2 = np.logical_and( pat, data['af'] < frequency_common )
                        rare = np.logical_or.reduce( (var1, var2) )
                        common = np.logical_and.reduce( ( keep, data['af'] >= frequency_common ) )
                        if np.any(rare):
                            update_variants_per_gene(data[samples].loc[rare], bool_single_rare, bool_multi_rare, samples, gene_old)
                        if np.any(common):
                            update_gc_per_gene(data[samples].loc[common], data[['id','chrom','ref','alt','start','end']].loc[common], 'hetero', bool_gc_hetero, fout_gc_hetero, samples, gene_old)
                            update_gc_per_gene(data[samples].loc[common], data[['id','chrom','ref','alt','start','end']].loc[common], 'homo', bool_gc_homo, fout_gc_homo, samples, gene_old)
                        n_genes += 1
                        print('Gene {}; number of genes: {}'.format(gene_old, n_genes))
                    gene_data = []
                    gene_old = gene
                gene_data.append([idd, chrom, start, end, ref, alt, gene, feat_type, cons, af, clin_sig])
        l = fin_ann.readline()

data = pd.DataFrame(bool_single_rare)
data.to_csv('data_al1_rare.csv')
print('Number of mutated samples for each gene [single rare]')
print(data.sum(axis = 0).sort_values())
data = pd.DataFrame(bool_multi_rare)
data.to_csv('data_al2_rare.csv')
print('Number of mutated samples for each gene [multi rare]')
print(data.sum(axis = 0).sort_values())
data = pd.DataFrame(bool_gc_homo)
data.to_csv('data_gc_homo.csv')
data = pd.DataFrame(bool_gc_hetero)
data.to_csv('data_gc_hetero.csv')

fout_gc_homo.close()
fout_gc_hetero.close()

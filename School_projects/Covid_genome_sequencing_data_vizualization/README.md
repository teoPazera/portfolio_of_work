# Covid_data_vizualization
group project analyzing data from genome sequencing of covid patients from january/february 2021
assignment of a project was:
COVID project for Data visualization course 2022/23

This folder contains data from three sequencing runs (sekvenačné behy), each run containing data for up to 
24 SARC-CoV-2 patient samples (pacientské vzorky).

* batch10 is a run from 2021-01-29, 24 samples 01-24
* batch11 is a run from 2021-02-03, 24 samples 01-24
* batch12 is a run from 2021-02-05, 23 samples 01-19,21-24


HOW SEQUENCING WORKS

Sequencing and its analysis consists of several phases:

* A sample contains a mixture of RNA from the SARS-CoV-2 virus, the human host and other bacteria or viruses. RNA is converted to DNA and SARS-CoV-2 DNA is copied many times using a technology called PCR. However, sometimes the sample is of lower quality or contains mutations in positions needed by PCR, and therefore we get only a small amount of viral DNA and perhaps higher fraction of human DNA or other contamination.

* A specific DNA string called barcode is attached to both ends of each molecule in a sample and 24 samples are then pooled together and sequenced. In this case sequencing is done using Oxford Nanopore technology. The result are so called reads (čítania), strings of letters A,C,G,T.

* In this dataset, reads have typical length 1700-2000. Ideally they would exactly correspond to some molecule in the sample, but in reality they contain errors. Up to cca 10% of letters in a read may be erroneous.

* Using the barcodes we recognize which reads come from each sample. Then we compare the reads to the reference SARS-CoV-2 genome and find from which location they come. For each sample, we want to have at least 50 reads from each location in the reference genome, ideally much more. Then we search if there are systematic differences between the reads and the reference genome; these correspond to mutations. (A difference which occurs only in one read is likely an error, a difference which occurs in the majority of reads is likely a mutation).


DESCRIPTION OF THE FILES

For each batch there are three files:

(A) File batchXX-results.tsv contains the list of samples with five columns:

  (1) identifier of the sample (barcode number 01-24),
  (2) the number of bases that could not be determined in the sample
  (3) the number of mutations found in the sample
  (4) the Pango lineage of the sample
  (5) the probability that the lineage was determined correctly

Column 2 has always some undetermined bases because the very start and end of the virus cannot be sequenced. But if the number is higher than 200, it meas that some parts that can be sequenced, were not sufficently covered by reads.

The lineage specifies the variant of the virus (https://cov-lineages.org/). 
From the limited data that we have it seems that B.1.258 was highly prevalent in Slovakia 
towards the end of 2020 and was replaced by B.1.1.7 in early 2021. B.1.1.7 is known as the alpha variant. 
The alpha variant had a higher number of mutations that other variants circulating at that time (column 3).

(B) File batchXX-reads.tsv contains the list of reads from the whole sequencing run with the following columns:

   (1) ID of the read (useful only for connecting with file C)
   (2) time when the read started sequencing
   (3) length of the read (how many A,C,G,T letters)
   (4) estimated quality of the read (higher number should mean fewer errors)
   (5) barcode I (see below)
   (6) barcode II
   (7) which organism it comes from
   (8) barcode III
   (9) status code

There are three barcodes. The first is assigned if at least one end of the reads has a recognizable barcode, otherwise it is "unclassified". The second is stricter and it is assigned if both ends have the same recognizable barcode, otherwise "unclassified". Finally the third one indicates if the read was actually used to search for mutations. Here we require that it has the stricter barcode II plus satisfies additional requirements, such as appropriate length and sufficient quality. Otherwise it has value "none".

Organisms in column 7 are "target" (SARS-CoV-2), "human", "bacteria", "virus" (other than SARS-CoV-2) or "none" if the read does not match anything in our database.

Status code is the classification of the read to some error classes, where class "G.ok" meas the read was used for finding mutations. However this classification is imperfect because some reads may have several issues but only one is chosen. It was used to generate Fig 2 in the article https://doi.org/10.1371/journal.pone.0259277

(C) File batchXX-match.tsv contains list of similarities found between the reads and the SARS-CoV-2 reference genome. It has the following columns:

  (1) The number of matching letters
  (2) Which of the two DNA strands is the read coming from
  (3) ID of the read (can be used to join with the file B)
  (4) length of the read (how many A,C,G,T letters)
  (5) where the matching region starts on the read
  (6) where the matching region ends on the read
  (7) ID of the reference sequence (always MN908947.3)
  (8) length of the reference sequence (always 29903 letters)
  (9) where the matching region starts on the reference
 (10) where the matching region ends on the reference
 (11) percentage of identical bases within the matching region (0-100%)
 
This file can be useful for studying how many reads cover individual parts of the reference and how good is the match between the read and the reference (higher quality reads should have a higher percentage if identical bases in the last column).

Some reads have multiple lines in this file. This happens when different parts of a read match to different parts of the reference genome. On the other hand, reads that do not match SARS-CoV-2 genome at all have no line in this file.


POSSIBLE QUESTIONS TO STUDY

* Compare the frequency of problematic reads (such as short / long / low quality / non-target) among samples in a batch and 
among batches. Which of these problems tend to happen together in the same read? 

* Is the estimated quality of the read related to it percent of matching bases?  How many reads with unknown source have also 
low quality or very small length?

* How many reads cover individual parts of the reference genome (from file C)? Is the number of bases with low coverage 
really related to the number of bases that could not be determined in file A? Is the average length or quality of reads 
different in different parts of the reference genome?

* Are any of the characteristics changing during the sequencing run (using the time column from file B)?

* Are there any differences between the alpha variant (B.1.1.7) samples and other variants?


MORE INFORMATION

* Brona Brejova would be happy to answer your questions.

* Article https://doi.org/10.1371/journal.pone.0259277 contains some analysis of this data. These three runs are denoted UKBA-10, UKBA-11 and UKBA-12 in this article.

* A more detailed description of the sequencing process can be found in article https://ceur-ws.org/Vol-2962/paper11.pdf

RNA
====

The emergence of single-cell profiling technologies has revolutionized our ability to acquire
high-resolution data, enabling the characterization of individual cells at distinct developmental states.
However, a significant challenge arises because cells are typically destroyed during measurement, allowing
only statistical data collection for individual samples at particular time steps. This approach prevents
the preservation of temporal correlations and limits access to ground-truth trajectories of cellular processes.


Understanding the time evolution of cellular processes, especially in response to external stimuli, is a
fundamental question in biology. To address this, we deploy \texttt{JKOnet\textsuperscript{$\ast$}} to
analyze single-cell RNA sequencing (scRNA-seq) data from embryoid bodies, capturing the differentiation
of human embryonic stem cells over a period of 27 days. By investigating
\texttt{JKOnet\textsuperscript{$\ast$}}'s ability to predict the evolution of cellular
and molecular processes, we aim to overcome the challenge of unaligned distributions across
snapshots and gain insights into the dynamics of cell differentiation.

.. toctree::
   :maxdepth: 2

   tutorial_rna
   results_rna
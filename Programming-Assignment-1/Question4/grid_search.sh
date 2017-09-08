#!/bin/bash

dataroot="../../../../CS6510/PA-1"

upoly=10
lpoly=1
njobs=2

for i in ` seq $lpoly $njobs $upoly `;
	do
		for j in `seq 1 1 $( expr $njobs - 1 ) `;
			do
				echo "Job Started with q = $( expr $i + $j - 1 )"
				python3 Main_single_kernel.py --dataroot $dataroot --kernel polynomial --q $( expr $i + $j - 1) --normalize &
			done
		echo "Job Started with q = $( expr $i + $njobs - 1)"
		python3 Main_single_kernel.py --dataroot $dataroot --kernel polynomial --q $( expr $i + $njobs - 1 ) --normalize
	done
	
ugauss=5
lgauss=0.5
step=0.5
njobs=2
iseq_step=$(python3 -c "print($step * $njobs)")
jseq_end=$(python3 -c "print($step * $njobs - $step)")

for i in ` seq $lgauss $iseq_step $ugauss `;
	do
		for j in ` seq $step $step $jseq_end `;
			do
				job_id=$(python3 -c "print($i + $j - $step)")
				echo "Job Started with sigma = $job_id"
				python3 Main_single_kernel.py --dataroot $dataroot --kernel gaussian --sigma $job_id --normalize &
			done
		
		job_id=$(python3 -c "print($i + $jseq_end)")
		echo "Job Started with sigma = $job_id"
		python3 Main_single_kernel.py --dataroot $dataroot --kernel gaussian --sigma $job_id --normalize
	done

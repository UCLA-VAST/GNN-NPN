mkdir -p results

for b in 6 8 10 12 14 16
do
    python main.py uniquify-truth-tables configs/epfl_6bit.yaml \
        dataset.0.name=epfl_${b}bit \
        dataset.0.num_bit=${b} \
        npn.truth_table_path=results/epfl_truth_tables_${b}bit.txt \
        npn.indices_path=results/epfl_unique_indices_${b}bit.txt \
        $@
done
for b in 6 8 10 12 14 16
do
    python prepare_data.py rewrite_aig --cut-size ${b}
done
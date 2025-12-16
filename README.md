# Example Usages

## SASA.py
python SASA.py sim_output --standardize-by-chain-length --placement-combos --combine-plots --frame-time-ps 50 --verbose
python SASA.py sim_output --standardize-by-chain-length --placement-combos --combine-plots --sys-type combo --verbose --chunked-chains

## rdf.py
python rdf.py sim_output --sys-type combo --placement-combos --chunked-chains --traj-cap 2

## dist-vs-angle-h2o.py
python dist-vs-angle-h2o.py /home/lucentlabs/tsavitski/FisH/small_sim_output/ --chunked-chain --sys-type solo --angle-min 104.38 --angle-max 104.61 --dist-max 15000 --standardize-by-chain-length --output solo-sys-water-angle.png

python dist-vs-angle-h2o.py /home/lucentlabs/tsavitski/FisH/sim_output/ --chunked-chain --sys-type combo --angle-min 104.38 --angle-max 104.61 --dist-max 15000 --standardize-by-chain-length --output combo-sys-water-angle.png
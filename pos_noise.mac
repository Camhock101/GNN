# Watchman example macro
/glg4debug/glg4param omit_muon_processes  0.0
/glg4debug/glg4param omit_hadronic_processes  0.0

/rat/db/set DETECTOR experiment "Watchman"
/rat/db/set DETECTOR geo_file "Watchman/Watchman1.geo"
#/rat/db/set WATCHMAN_PARAMS photocathode_coverage 0.20

/run/initialize

#/tracking/storeTrajectory 1
#/tracking/FillPointCont 1

/rat/proc noise
/rat/procset rate 3000.0
/rat/procset nearhits 0
/rat/procset lookback 50.0
/rat/procset lookforward 200.0

# BEGIN EVENT LOOP
/rat/proc lesssimpledaq
/rat/proc count
/rat/procset update 1000

# Use IO.default_output_filename
/rat/proclast outroot
#END EVENT LOOP

/generator/add combo spectrum:regexfill:poisson
/generator/vtx/set e+ flat
/generator/pos/set fill_vol
/generator/rate/set 1.0

/run/beamOn 1000



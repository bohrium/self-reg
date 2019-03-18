plots-gauss:
	python vis.py experdata_gauss.txt gradstats_gauss.txt OUT-GD   plots/gauss/out-gd.png   0.025
	python vis.py experdata_gauss.txt gradstats_gauss.txt OUT-SGD  plots/gauss/out-sgd.png  0.025
	python vis.py experdata_gauss.txt gradstats_gauss.txt OUT-DIFF plots/gauss/out-diff.png 0.025
	python vis.py experdata_gauss.txt gradstats_gauss.txt GEN-GD   plots/gauss/gen-gd.png   0.025
	python vis.py experdata_gauss.txt gradstats_gauss.txt GEN-SGD  plots/gauss/gen-sgd.png  0.025

plots-death: experdata_death.txt gradstats_death.txt
	python vis.py experdata_death.txt gradstats_death.txt OUT-GD   plots/death/out-gd.png   0.0004
	python vis.py experdata_death.txt gradstats_death.txt OUT-SGD  plots/death/out-sgd.png  0.0004
	python vis.py experdata_death.txt gradstats_death.txt OUT-DIFF plots/death/out-diff.png 0.0004
	python vis.py experdata_death.txt gradstats_death.txt GEN-GD   plots/death/gen-gd.png   0.0004
	python vis.py experdata_death.txt gradstats_death.txt GEN-SGD  plots/death/gen-sgd.png  0.0004

plots-logis: experdata_logis.txt gradstats_logis.txt
	python vis.py experdata_logis.txt gradstats_logis.txt OUT-GD   plots/out-gd.png   0.03
	python vis.py experdata_logis.txt gradstats_logis.txt OUT-SGD  plots/out-sgd.png  0.03
	python vis.py experdata_logis.txt gradstats_logis.txt OUT-DIFF plots/out-diff.png 0.03
	python vis.py experdata_logis.txt gradstats_logis.txt GEN-GD   plots/gen-gd.png   0.03
	python vis.py experdata_logis.txt gradstats_logis.txt GEN-SGD  plots/gen-sgd.png  0.03

extended:
	python vis.py experdata_death.txt gradstats_death.txt OUT-SGD  plots/out-sgd.extended.png  1.000

### Usage
Extract the vcglib-1.0.1.zip to `Poisson_sample` as this file structure:
```angular2html
--prepare_data
----Poisson_sample
------vcglib-1.0.1
```

Then compile the files:
```shell
cd prepare_data/Poisson_sample
cmake .
make
```

Change `[num_points]` to the sampled number of interest:
```angular2html
./PdSampling [num_points] xxx.off xxx.xyz
```


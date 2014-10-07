dfresampling
============

Cython implementation of the image resampling method described in "On resampling of Solar Images", C.E. DeForest, Solar Physics 2004

Instructions
------------

Install the required dependencies:

```pip install -r requirements.txt```

Compile the Cython extension:

```python setup.py build_ext --inplace```

Then run the test program:

```python dfresampling/test.py```

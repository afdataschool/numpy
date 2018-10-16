---
title: "Operations on NumPy arrays"
teaching: 15
exercises: 15
questions:
- ""
objectives:
- "Explain how GitHub Pages produce web sites from Git repositories."
- "Explain Jekyll's formatting rules."
- "Explain the difference between element-wise and matrix product of two arrays"
- "Apply reduction functions (mean, min, max) along a given axis"
- "Find a specialised numerical algorithm from the ones available in numpy"
- "Sort arrays along given axis"

Multiplication of two arrays is elementwise. For example, to calculate a square of each element we may use:

~~~
a = np.arange(3)
b = a * a
~~~
{: .language-python}

Matrix inner products are calculated using the `np.dot` function:

~~~
np.dot(a, a)
5
~~~
{: .language-python}

For 1-D arrays the same result can be obtained by:

~~~
np.sum(a * a)
5
~~~
{: .language-python}


### Axis-based reductions

The `np.sum` function sums all elements regardless of the number of array dimensions:


~~~
b = np.arange(9).reshape(3,3)
np.sum(b)
~~~
{: .language-python}

If you want to sum only columns or rows, you need to pass the index of the axis over which you want to sum:

~~~
np.sum(b, 0)
np.sum(b, 1)
~~~
{: .language-python}

Other similar reduction functions are `np.min`, `np.max` or `np.mean`:

~~~
np.min(b)
np.min(b, 0)
np.min(b, 1)
~~~
{: .language-python}

You can also find the index of the minimum element in each axis:

~~~
np.argmin(b, 0)
array([0, 0, 0])
~~~
{: .language-python}

### Sorting

NumPy also implement various sorting algorithms. To sort elements of an array you can use `np.sort` functions:

~~~
a = np.random.rand(4)
a
array([ 0.9490829 ,  0.07528673,  0.17463988,  0.95964801])
np.sort(a)
array([ 0.07528673,  0.17463988,  0.9490829 ,  0.95964801])
~~~
{: .language-python}

Similarly to the reduction functions, you can also pass the axis index to sort along: 

~~~
b = a.reshape(2, 2)
np.sort(b, 0)
np.sort(b, 1)
~~~
{: .language-python}

`np.argsort` returns the order of elements in a sorted array:

~~~
np.argsort(a)
~~~
{: .language-python}

> ## Special modules
>
> NumPy also provides extra modules implementing basic numerical methods:
> 
> * `np.linalg` -- linear algebra,
> * `np.fft` -- fast Fourier transform,
> * `np.random` -- random number generators.
{: .callout}

> ## Finding closest element
>
> Generate a 10 x 3 array of random numbers (using `np.random.rand`). From each row, find the column index of the element 
> closest to 0.75. Make use of np.abs and np.argmin. The result should be a one-dimensional array of integers from 0 to 2.
{: .challenge}

> ## Solving linear equations 
>
> Solve the following system of linear equations using `np.linalg.solve`. Test the solution.
> $$2x + 3y = 3$$
> $$5x - y = 6$$
{: .challenge}


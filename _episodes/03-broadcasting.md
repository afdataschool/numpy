---
title: "Broadcasting"
teaching: 40
exercises: 20
questions:
- ""
objectives:
- "add a scalar to all elements of an array"
- "predict the result of additions of matrices and row or column vectors"
- "explain why broadcasting is better than using for loops"
- "understand the rules of broadcasting and can predict the shape of broadcasted arrays"
- "control broadcasting using `np.newaxis` object"
- "Explain overall organization of lesson files."
keypoints:
- ""
---

It’s possible to do operations on arrays of different sizes. In some case NumPy can
transform these arrays automatically so that they all have the same size: this conversion is
called **broadcasting**.

![numpy broadcasting in 2D](fig/numpy_broadcasting.png "numpy broadcasting in 2D")

Let's try to reproduce the above diagram. First, we create two one-dimensional arrays:

~~~
a = np.arange(4) * 10
b = np.arange(3)
~~~
{: .language-python}

We can tile them in 2D using `np.tile` function:

~~~
b2 = np.tile(b, (4, 1))
~~~
{: .language-python}

We do the same with the second array, but we need also to transpose (exchange columns with rows) the resulting array:

~~~
a2 = np.tile(a, (3, 1))
a2 = a2.T
a2
~~~
{: .language-python}

Note that the `np.tile` function creates new arrays and copies the data. Then you can add the arrays element-wise:

~~~
a2 + b2
~~~
{: .language-python}

In the second example we add a one-dimensional array to a two-dimensional array. NumPy will automatically "tile" the 1D array along the missing direction:

~~~
a2 + b
array([[ 0,  1,  2],
       [10, 11, 12],
       [20, 21, 22],
       [30, 31, 32]])
~~~
{: .language-python}

However, in this case no copy of `b` array is involved. NumPy will instead use the same data in `b` for each row of `a` -- we will cover the mechanism behind it at the end of the lesson. 

In the third example we add a single column with a single vector. To obtain a column array from a 1D array we need to convert it to 2D array of four rows and one column. In NumPy we can add singular dimensions (dimensions of size 1) by a special object `np.newaxis`:

~~~
a.shape
(4,)
a_column = a[:, np.newaxis]
a_column.shape
(4, 1)
a_column
array([[ 0],
       [10],
       [20],
       [30]])
~~~
{: .language-python}

We can add a column vector and a 1D array:

~~~
a_column + b
array([[ 0,  1,  2],
       [10, 11, 12],
       [20, 21, 22],
       [30, 31, 32]])
~~~
{: .language-python}

This is the same as adding a column and  row vector:

```
>>> b_row = b[np.newaxis, :]
>>> b_row
array([[0, 1, 2]])
>>> b_row.shape
(1, 3)
>>> a_column + b_row
array([[ 0,  1,  2],
       [10, 11, 12],
       [20, 21, 22],
       [30, 31, 32]])
```

> ## Normalising data {.challenge}
>
> Given the following array:
>
> ```
> a = np.random.rand(10, 100) 
> ```
> 
> For each column of `a` subtract its mean. Next, do the same with rows.


Broadcasting seems a bit magical, but it is actually quite natural to use it when we want to solve a problem whose output data is an array with more dimensions than input data. There a simple rule that allow to determine the validity of broadcasting and the shape of broadcasted arrays:

>  In order to broadcast, the size of the trailing axes for both arrays in an operation must either be the same or one of them must be one. 

This does indeed work for the three addition from the figure

```
a:      4 x 3 
b:      4 x 3
result: 4 x 3

a:      4 x 3
b:          3
result: 4 x 3

a:      4 x 1
b:          3
result: 4 x 3
```

Lets look at two more examples:

```
Image  (3d array): 256 x 256 x 3
Scale  (1d array):             3
Result (3d array): 256 x 256 x 3

A      (4d array):  8 x 1 x 6 x 1
B      (3d array):      7 x 1 x 5
Result (4d array):  8 x 7 x 6 x 5
```

> ## Broadcasting rules {.challenge}
> 
> Given the arrays:
> ```
> X = np.random.rand(10,3)
> Y = np.random.rand(3)
> ```
> 
> which of the following will *not* produce an error:
> 
> a) `X + Y`
> 
> b) `X[np.newaxis, :] + Y`
> 
> c) `X + Y[:, np.newaxis]`
> 
> d) `X[:, np.newaxis] + Y`
> 
> e) `X + Y[np.newaxis, :]`
>
> f) `X[:, np.newaxis, :] + Y`
> 
> What will be the shapes of the final broadcasted arrays? Try to guess and then check.

> ## Three-dimensional broadcasting {.challenge}
>
> Below, produce the array containing the sum of every element in `x` with every element in `y`
>
> ```python
> x = np.random.rand(3, 5)
> y = np.random.randint(10, size=8)
> z = x # FIX THIS
> ```

> ## Broadcasting indices {.challenge}
>
> Predict and verify the shape of `y`:
> 
> ```python
> x = np.empty((10, 8, 6))
> 
> idx0 = np.zeros((3, 8)).astype(int)
> idx1 = np.zeros((3, 1)).astype(int)
> idx2 = np.zeros((1, 1)).astype(int)
> 
> y = x[idx0, idx1, idx2]
> ```


A lot of grid-based or network-based problems can also use broadcasting. For instance, if we want to compute the distance from the origin of points on a 10x10 grid, we can do
```
>>> x = np.arange(5)
>>> y = np.arange(5)[:, np.newaxis]
>>> distance = np.sqrt(x ** 2 + y ** 2)
>>> distance
array([[ 0.        ,  1.        ,  2.        ,  3.        ,  4.        ],
       [ 1.        ,  1.41421356,  2.23606798,  3.16227766,  4.12310563],
       [ 2.        ,  2.23606798,  2.82842712,  3.60555128,  4.47213595],
       [ 3.        ,  3.16227766,  3.60555128,  4.24264069,  5.        ],
       [ 4.        ,  4.12310563,  4.47213595,  5.        ,  5.65685425]])
```

> ## Creating a two-dimensional grid {.challenge}
> 
> What are the dimensionalities of `x`, `y` and `z` in the two cases:
>
> ```
> x, y = np.mgrid[:10, :5]
> z = x + y
> ```
> 
> and 
> 
> ```
> x, y = np.ogrid[:10, :5]
> z = x + y
> ```
> 
> What might be the advantage of using `np.ogrid` over `np.mgrid`?

> ## Worked example: Route 66 {.callout}
>
> Let’s construct an array of distances (in miles) between cities of Route 66: Chicago, Springfield, Saint-Louis, Tulsa, Oklahoma City, Amarillo, Santa Fe, Albuquerque, Flagstaff and Los Angeles.
> ```
> >>> mileposts = np.array([0, 198, 303, 736, 871, 1175, 1475, 1544,
> ...        1913, 2448])
> >>> distance_array = np.abs(mileposts - mileposts[:, np.newaxis])
> >>> distance_array
> array([[   0,  198,  303,  736,  871, 1175, 1475, 1544, 1913, 2448],
>        [ 198,    0,  105,  538,  673,  977, 1277, 1346, 1715, 2250],
>        [ 303,  105,    0,  433,  568,  872, 1172, 1241, 1610, 2145],
>        [ 736,  538,  433,    0,  135,  439,  739,  808, 1177, 1712],
>        [ 871,  673,  568,  135,    0,  304,  604,  673, 1042, 1577],
>        [1175,  977,  872,  439,  304,    0,  300,  369,  738, 1273],
>        [1475, 1277, 1172,  739,  604,  300,    0,   69,  438,  973],
>        [1544, 1346, 1241,  808,  673,  369,   69,    0,  369,  904],
>        [1913, 1715, 1610, 1177, 1042,  738,  438,  369,    0,  535],
>        [2448, 2250, 2145, 1712, 1577, 1273,  973,  904,  535,    0]])
>```
> ![Distances on Route 66](fig/route66.png)

> ## Distances {.challenge}
> 
> Given an array of latitudes and longitudes of major European capitals calculate pairwise distances between them. Use the approximate formula: 
>
> $$D=6371.009\sqrt{(\Delta\phi)^2 + (\Delta\lambda)^2}\qquad \text{(in kilometers)},$$
>
> where $\Delta\phi=\phi_1-\phi_2$ and $\Delta\lambda=\lambda_1-\lambda_2$ are the differences between the latitudes and longitude of two cities in radians. (*Hint*: To convert degrees to radians multiply them by $\pi/180$).
> ```
> coords = np.array([
>                   [ 23.71666667,  37.96666667], # Athens
>                   [ 13.38333333,  52.51666667], # Berlin
>                   [ -0.1275    ,  51.50722222], # London
>                   [ -3.71666667,  40.38333333], # Madrid
>                   [  2.3508    ,  48.8567    ], # Paris
>                   [ 12.5       ,  41.9       ]  # Rome
                    ]) 
> ```
> When you are done you can compare the results with a more [precise formula](https://en.wikipedia.org/wiki/Geographical_distance#Spherical_Earth_projected_to_a_plane):
>
> $$D=6371.009\sqrt{(\Delta\phi)^2 + (\cos(\phi_m)\Delta\lambda)^2}$$
>
> where $\phi_m = (\phi_1+\phi_2) / 2$ is the mean latitude.




Each lesson is made up of *episodes*, which are focused on a particular topic and
include time for both teaching and exercises.
The episodes of this lesson explain the tools we use to create lessons
and the formatting rules those lessons must follow.

> ## Why "Episodes"?
>
> We call the parts of lessons "episodes" because
> every other term (like "topic") already has multiple meanings,
> and because it encourages us to think of breaking up our lessons
> into chunks that are about as long as a typical movie scene,
> which is better for learning than long blocks without interruption.
{: .callout}

Our lessons need artwork,
CSS style files,
and a few bits of Javascript.
We could load these from the web,
but that would make offline authoring difficult.
Instead, each lesson's repository is self-contained.

The diagram below shows how source files and directories are laid out,
and how they are mapped to destination files and directories:

![Source and Destination Files]({{ page.root }}/fig/file-mapping.svg)

> ## Collections
>
> As described [earlier]({{ page.root }}/02-tooling/#collections),
> files that appear as top-level items in the navigation menu are stored in the root directory.
> Files that appear under the "extras" menu are stored in the `_extras` directory,
> while lesson episodes are stored in the `_episodes` directory.
{: .callout}

## Helper Files

As is standard with [Jekyll][jekyll] sites,
page layouts are stored in `_layouts`
and snippets of HTML included by these layouts are stored in `_includes`.
Each of these files includes a comment explaining its purpose.

Authors do not have to specify that episodes use the `episode.html` layout,
since that is set by the configuration file.
Pages that authors create should have the `page` layout unless specified otherwise below.

The `assets` directory contains the CSS, Javascript, fonts, and image files
used in the generated website.
Authors should not modify these.

# Standard Files

When the lesson repository is first created,
the initial author should create a `README.md` file containing
a one-line explanation of the lesson's purpose.

The [lesson template]({{ site.template_repo }}) provides the following files
which should *not* be modified:

*   `CONDUCT.md`: the code of conduct.
*   `LICENSE.md`: the lesson license.
*   `Makefile`: commands for previewing the site, cleaning up junk, etc.

## Starter Files

The `bin/lesson_initialize.py` script creates files that need to be customized for each lesson:

`CONTRIBUTING.md`
:   Contribution guidelines.
    The `issues` and `repo` links at the bottom of the file must be changed
    to match the URLs of the lesson:
    look for uses of `FIXME`.

`_config.yml`
:   The [Jekyll][jekyll] configuration file.
    This must be edited so that its links and other settings are correct for this lesson.
    *   `carpentry` should be either "dc" (for Data Carpentry) or "swc" (for Software Carpentry).
    *   `title` is the title of your lesson,
        e.g.,
        "Defence Against the Dark Arts".
    *   `email` is the contact email address for the lesson.

`CITATION`
:   A plain text file explaining how to cite this lesson.

`AUTHORS`
:   A plain text file listing the names of the lesson's authors.

`index.md`
:   The home page for the lesson.
    1.  It must use the `lesson` layout.
    2.  It must *not* have a `title` field in its [YAML][yaml] header.
    3.  It must open with a few paragraphs of explanatory text.
    4.  That introduction must be followed by a single `.prereq` blockquote
        detailing the lesson's prerequisites.
        (Setup instructions appear separately.)
    5.  That must be followed by inclusion of `syllabus.html`,
        which generates the syllabus for the lesson
        from the metadata in its episodes.

`reference.md`
:   A reference guide for the lesson.
    The template will automatically generate a summary of the episodes' key points.
    1.  It must use the `reference` layout.
    2.  Its title must be `"Reference"`.
    3.  Its permalink must be `/reference/`.
    4.  It should include a glossary, laid out as a description list.
    5.  It may include other material as appropriate.

`setup.md`
:   Detailed setup instructions for the lesson.
    Note that we usually divide setup instructions by platform,
    e.g.,
    include level-2 headings for Windows, macOS, and Linux
    with instructions for each.
    The [workshop template]({{ site.workshop_repo }})
    links to the setup instructions for core lessons.
    1.  It must use the `page` layout.
    2.  Its title must be `"Setup"`.
    3.  Its permalink must be `/setup/`.
    4.  It should include whatever setup instructions are required.

`_extras/about.md`
:   General notes about this lesson.
    This page includes brief descriptions of Software Carpentry and Data Carpentry,
    and is a good place to put institutional acknowledgments.

`_extras/discussion.md`
:   General discussion of the lesson contents for learners who wish to know more:
    This page normally includes links to further reading
    and/or brief discussion of more advanced topics.
    1.  It must use the `page` layout.
    2.  Its title must be `"Discussion"`.
    3.  Its permalink must be `/discuss/`.
    4.  It may include whatever content the author thinks appropriate.

`_extra/figures.md` and `_includes/all_figures.html`
:   Does nothing but include `_includes/all_figures.html`,
    which is (re)generated by `make lesson-figures`.
    This page displays all the images referenced by all of the episodes,
    in order,
    so that instructors can scroll through them while teaching.

`_extras/guide.md`
:   The instructors' guide for the lesson.
    This page records tips and warnings from people who have taught the lesson.
    1.  It must use the `page` layout.
    2.  Its title must be `"Instructors' Guide"`.
    3.  Its permalink must be `/guide/`.
    4.  It may include whatever content the author thinks appropriate.

## Figures

All figures related with the lesson **must** be placed inside the directory `fig` at the root of the project.

{% include links.md %}

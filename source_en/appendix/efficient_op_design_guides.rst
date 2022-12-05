=================================================================
Design Guide for Efficient Operators(ONNX)
=================================================================

When the range of operator design matches the range of hardware support, the hardware performance can be exploited more fully to improve the model inference speed.
This section explains how to implement efficient design algorithms on the ``AX620`` hardware platform.

----------------------------------
Convolution
----------------------------------

.. note::

    The convolution operator consumes an input tensor and a filter, and computes the output.

**Conv** Supported ``OpSet Version``: ``1``, ``11-13``

.. csv-table::
   :file: ../csv/Conv_OP.csv
   :header-rows: 1

.. hint::

  ``input/output_channel``
      - Most efficient when ``input_channel`` is a multiple of ``16`` and ``output_channel`` is a multiple of ``8``
      - When the multiplier limit is not met, the calculation is wasted to the corresponding multiplier.

----------------------------------
ConvTranspose
----------------------------------

``ConvTranspose`` has the most efficient support for the following three cases.

* kernel_size is ``2 x 2``, stride is ``2``, pad takes ``0``
* kernel_size is ``4 x 4``, stride is ``2``, pad takes ``1``
* kernel_size is ``4 x 4``, stride is ``4``, pad takes ``0``

.. attention::

    The efficiency of ``ConvTranspose`` is slightly lower than that of the ``resize`` operator, which performs the same upsampling function.

----------------------------------
Linear
----------------------------------

It is recommended that ``channels`` be a multiple of ``16``.

----------------------------------
Activation
----------------------------------

- ``ReLU`` has the most efficient support
- ``LeakyReLU``, ``HardSwish``, ``Swish``, ``Mish`` are also efficiently supported (but weaker than ``ReLU``)
- ``PReLU`` support is less efficient

----------------------------------
Transpose/Reshape
----------------------------------

.. attention::

    The implementation is inefficient and should be avoided.

----------------------------------
Pool
----------------------------------

.. list-table::
    :widths: 10 60
    :header-rows: 1

    * - Operator
      - Efficient suggestions

    * - MaxPool
      - Efficient support for the case ``kernel_size <= 2`` and ``kernel_size == stride``, it is recommended to try to ``kernel_size <= 3``
    
    * - AvgPool
      - ``kernel_size`` to the power of ``2`` is the most efficient, and it is recommended that the maximum exceed ``32``.

----------------------------------
Resize
----------------------------------

- ``scale`` only supports powers of two, suggested in the range [1/16, 1/8, 1/4, 1/2, 2, 4, 8, 16].
- ``mode`` only supports ``nearest``, ``bilinear`` and ``area``.
  
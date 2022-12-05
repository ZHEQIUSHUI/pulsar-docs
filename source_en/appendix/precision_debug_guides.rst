==================================================================================
Accuracy loss troubleshooting and accuracy tuning suggestions
==================================================================================

******************************
Accuracy Loss Identification
******************************

.. attention::

    If there is a loss of accuracy in the converted model, please follow the recommended way to troubleshoot the ``stage`` or ``layer`` that is causing the problem
    
.. _checklists:

------------------------
CheckLists
------------------------

.. data:: The first step is to specify the hardware platform on which the accuracy loss occurs

    1. only on the ``AX`` platform

        ::

            Please continue down the list.
    
    2. All other platforms have accuracy loss occurs

        ::

            Common problem, users need to decide whether to train a better model and then re-quantize it;
            Determine if other platforms use INT8 or INT16 quantization, or a mix of quantization.

.. data:: In the second step, determine the stage where the accuracy loss occurs

    1. ``pulsar run`` has a low accuracy (``cos-sim < 98%``)

        ::

            Please follow the [Step 3] recommendations to continue the investigation

    2. The upper board is connected to the user's ``post-processing`` program, and the accuracy is very low after parsing

        ::

            Please follow [step 4] and continue to the next step


.. _three_step:

.. data:: Step 3, cos-sim is below 98%, troubleshooting suggestions

    1. the ``output_config.prototxt`` file required for ``pulsar run`` **must** be generated automatically by ``pulsar build``
    2. check that the ``color space`` and ``mean/std`` configurations in the ``config.prototxt`` configuration file are correct
    3. use ``pulsar run`` to compare the ``cos-sim`` values between ``model.lava_joint`` and ``model.onnx`` to see if accuracy loss occurs
    4. Use layer-by-layer splitting to see the ``layer`` where the loss of precision occurs

.. _four_step:

.. data:: Step 4, low accuracy on the board, troubleshooting suggestions

    1. When executing the ``run_joint`` command, it will print some information about the ``joint`` model, so you need to check if the ``post-processor`` is parsing the output data correctly.
    2. If other platforms don't drop points, but ``BadCase`` is reported on ``AX`` platform, see :ref:`Upboard accuracy loss troubleshooting method <precision_loss_on_board>`.

.. data:: Step 5, get help from AXera
    
    When the user still can't solve the problem after the first four steps, please send the relevant ``log`` and ``conclusion`` to ``FAE`` colleagues, so that ``AX`` engineers can locate the problem

----------------------------------------------
Accuracy loss occurs after model compilation
----------------------------------------------

This section elaborates on :ref:`CheckLists <checklists>` in :ref:`third_step <three_step>`.

.. hint::

    ``pulsar run`` is an integrated tool in the ``Pulsar`` toolchain for simulation and pairing, see :ref:`Simulation and pairing on x86 platforms <pulsar_run_sim>` for details.

If the original ``onnx`` model is compiled into a ``joint`` model, the ``cos-sim`` of the ``pulsar run`` is very low, which means that the converted model is losing accuracy and the problem needs to be investigated.

.. data:: config Configuration

    The ``config`` required for ``pulsar run`` is automatically generated from the ``pulsar build``.

    .. code-block:: python
        :linenos:

        # Note that the following command is not complete
        pulsar build --input model.onnx --config config.prototxt --output_config output_config.prototxt  ...
        pulsar run model.onnx model.joint --config output_config.prototxt  ...

.. data:: csc & mean/std

    ``color space convert, csc`` After configuration, you need to configure ``mean/std`` in channel order.

    .. code-block:: python
        :linenos:

        # Configure the input data color space of the compiled model as BGR
        dst_input_tensors {
            color_space: TENSOR_COLOR_SPACE_BGR
        }

        # mean/std needs to be filled in the order of BGR
        input_normalization {
            mean: [0.485, 0.456, 0.406]  # mean
            std: [0.229, 0.224, 0.255]   # std
        }

    The ``color_space`` in ``dst_input_tensors`` is ``BGR``, which means that the calibration image data is read in ``BGR`` format at compile time, so that ``mean/std`` is also set in ``BGR`` order.

.. data:: check if the model has lost accuracy during the quantization phase

    During the compilation of ``pulsar build``, an intermediate file ``model.lava_joint`` is generated for debugging, which is passed through

    .. code-block:: python
        :linenos:
        
        # Note that the following commands are incomplete
        pulsar run model.onnx model.lava_joint --input ...

    You can verify that there is no loss of precision in the quantization phase.

.. data:: Model quantization stage lost accuracy solution

    1. add quantitative data sets

        .. code-block:: python
            :linenos:

            dataset_conf_calibration {
                path: "imagenet-1k-images.tar"
                type: DATASET_TYPE_TAR
                size: 256 # The actual number of data needed for calibration during compilation
                batch_size: 32 # default is 32, can be changed to other values
            }

    2. Adjustment of quantitative strategies and quantitative methods

        - Quantification strategy, ``CALIB_STRATEGY_PER_CHANNEL`` and ``CALIB_STRATEGY_PER_TENSOR``
        - quantization methods, ``OBSVR_METHOD_MIN_MAX`` and ``OBSVR_METHOD_MSE_CLIPPING``
        - Quantitative strategies and quantitative methods can be **two combinations**, where ``CALIB_STRATEGY_PER_CHANNEL`` may have dropped points
        - Recommend ``PER_TENSOR/MIN_MAX`` or ``PER_TENSOR/MSE_CLIPPING`` combinations

        .. code-block:: python
            :linenos:

            dataset_conf_calibration {
                path: "magenet-1k-images.tar" # quantified dataset
                type: DATASET_TYPE_TAR
                size: 256 # The actual number of data needed for calibration during compilation
                batch_size: 32 # default is 32, can be changed to other values

                calibration_strategy: CALIB_STRATEGY_PER_TENSOR # Quantification strategy
                observer_method: OBSVR_METHOD_MSE_CLIPPING # Quantification method
            }

    3. use ``INT16`` quantization

        - See :ref:`16bit quantization <Q16bit>` for details.
    
    4. turn on ``dataset_conf_error_measurement``, for error testing during compilation

        .. code-block:: python
            :linenos:

            dataset_conf_error_measurement {
                path: "imagenet-1k-images.tar"
                type: DATASET_TYPE_TAR
                size: 32
                batch_size: 8
            }

.. data:: Layer-by-layer comparison

    See :ref:`layer wise compare <layer_wise_compare>` for details.

.. data:: pulsar debug
    
    The ``pulsar debug`` function will be added later

.. _precision_loss_on_board:

----------------------------------------------
Accuracy loss occurs on board
----------------------------------------------

本节对 :ref:`CheckLists <checklists>` 中 :ref:`第四步 <four_step>` 进行详细说明.

This section details the :ref:`CheckLists <checklists>` in :ref:`fourth_step <four_step>`.

.. data:: Determining if the post-processor is wrong

    Using the ``run_joint`` command on the ``AX`` development board, you can implement board-side reasoning and then parse the results using the user's own postprocessor.

    To verify that the user's post-processor is error-free, you can compare the output of ``pulsar run`` with the output of ``run_joint`` for the same input condition, 
    
    Refer to the :ref:`gt folder comparison instructions <pulsar_run_gt_compare>`, if the comparison is successful, the **user's** postprocessor ``may`` have an error.

.. data:: The post-processor is correct, but the accuracy is still low.

    Possible reasons
        * ``npu simulator`` generated instructions and ``cmode`` ran inconsistent results.
        * ``run_joint.so`` and ``npu drive`` errors
    
    This kind of problem needs to be logged so that it can be fixed quickly.

.. data:: BadCase handling

    For this type of ``BadCase``, first check ``cos-sim`` with ``pulsar run``, if there is no serious point loss (below 98%), 
    
    Then send the ``BadCase`` to the board and run it with ``run_joint``,

    See if the results are consistent with ``pulsar run``, if not, it means there is a problem with the board and needs to be fixed by the ``AX`` engineer.

------------------------
Other notes
------------------------

If you need an ``AX`` engineer to troubleshoot the problem, please provide detailed log information and relevant experimental findings. 

>>> Note: If you can provide a minimum recurrence set, you can improve the efficiency of the problem.

.. note::

    In some cases the ``SILU`` function causes the ``mAP`` of the detection model to be very low, replacing it with the ``ReLU`` function will solve the problem.

.. note::

    If the ``quantized dataset`` is very different from the ``training dataset``, the accuracy will be significantly reduced.

    To determine whether the ``calibration`` choice is reasonable, you can select a ``pulsar run`` from the ``calibration`` dataset and perform a ``pulsar run`` to score it.

****************************
Precision tuning suggestions
****************************

For the quantized accuracy error, it is recommended that the user use the following ``2`` methods for optimization, both of which require reconversion of the model after configuration in the ``config.prototxt`` file.

--------------------
calibration settings
--------------------

* Two combinations of quantitative strategies and quantitative solutions
* Try to use other quantitative data sets
* Increase or decrease the amount of data as appropriate

--------------------
QAT Training
--------------------

When the accuracy of the model cannot be improved by using various tuning techniques, the model is probably the ``corner case`` of the ``PTQ`` scheme, and you can try to train it using ``QAT``.

.. attention::
    
    More tuning suggestions will be updated gradually.
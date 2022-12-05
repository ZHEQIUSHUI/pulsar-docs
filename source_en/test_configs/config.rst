.. _config_details:

============================
Configuration file details
============================

This section describes the ``Pulsar Config`` section.

------------------------------------
Overview of config.prototxt
------------------------------------

- ``Pulsar`` is a powerful and complex set of tools for compiling, simulating, and debugging models that often require the necessary configuration parameters to work exactly as intended
- Currently, all parameters can be passed to ``Pulsar`` through the configuration file, :ref: `A few parameters <some_params_called_by_cmdline>` can be specified or modified on an ad hoc basis through the command line parameter interface
- In addition to passing configuration parameters to the compiler, the configuration file also has an important role in guiding the compiler through the complex compilation process

.. mermaid::

  graph LR
    config_prototxt[config file]
    command_line[command line parameter]
    input_model[inputmodel-1<br/>inputmodel-2<br/>...<br/>input model-N] ----> super_pulsar[Pulsar]
    super_pulsar ----> output_model[output model]
    config_prototxt -.-> |Passing compilation parameters| super_pulsar
    command_line -.-> |Set and override compilation parameters| super_pulsar

.. attention::

    The compilation parameters passed through the command line parameter interface will override the parameters provided by the configuration file

**Content Format**

- The current configuration file format is a text format called ``prototxt``, whose contents can be read and modified directly with a text editor
- The configuration file in ``prototxt`` format can be commented internally, with comments starting with ``#``
- The version is ``proto3``

**Noun Conventions**

- Due to the complex structure of the configuration file and the depth of the configuration parameters, in order to facilitate the description and minimize the misunderstanding caused by improper terminology, we have agreed on the meaning of some common terms. If you find any unclear expressions, wording errors, etc., or have better suggestions in reading this series of documentation, please feel free to criticize and correct

**Parameter path**

- **Parameter path** is used to express the position of a configuration parameter in a multi-layer structure parameter
- When a configuration parameter is located in other structures with multiple nested levels, the name of the structure parameter at each level is used plus a dot ``. ``'' to express the location of the currently introduced parameter in the configuration file, e.g. the parameter represented by the string ``pulsar_conf.batch_size`` is located in the configuration file as follows:

  .. code-block:: sh
    :name: input_conf_items
    :linenos:
    
    pulsar_conf { # Compiler-related configuration
      batch_size: 1
    }

.. note::

  | Here **the configuration file itself is treated as an anonymous structure**, and the paths of its internal first-level parameters are the parameter names themselves
  | When introducing **generic data types** without parameter paths, because they can appear in multiple places in the configuration file
  | Some places use the full path or relative path of a parameter to express a parameter name

**Compilation Process**

- The compilation process is generally used as a proxy for compiling a model in one format into another format. For example, to compile a model in ``onnx`` format into ``joint`` format

**compile step**

- Compilation steps are generally used when a compilation process can be explicitly divided into several steps. For example, two ``onnx`` models are first compiled separately into ``joint`` format, and then the two ``joint`` models are fused to form a single ``joint`` model.
- When describing the configuration file, it may say "the entire compilation process is divided into three compilation steps, with configuration parameters for each **compilation step**"
- But when describing a particular **compile step** within a **compile process**, we may refer to the **compile step** as the **compile process** within a subsection, when the two terms refer to the same object. Note the contextual distinction

**compile parameters**

- Compilation parameters are used to refer to the parameters that need to be configured for a **compile process** or **compile step**.

-----------------------------------
config Internal structure overview
-----------------------------------

The ``config.prototxt`` consists of the following six sections, including:

- :ref:`input and output configuration <input_and_output_config>`
- :ref:`Select hardware platform <select_hardware>`
- :ref:`CPU subgraph settings <cpu_backend_settings>`
- :ref:`Special handling of Tensor <tensor_conf>`
- :ref:`Neuwizard configuration <neuwizard_conf>`
- :ref:`Configuration of Pulsar <pulsar_conf>`

config Internal structure example

.. code-block:: sh
  :name: config.prototxt outline
  :linenos:
  :emphasize-lines: 13-14, 17, 19
  
  # config.outline.prototxt

  # Basic input and output configuration
  input_path: # Relative path of the input model
  input_type: # Input model type, by default it is equal to INPUT_TYPE_AUTO, the compiler will infer the model file name automatically, but sometimes the inference result is not expected
  output_path: # The relative path of the output model
  output_type: # Output model type, if not specified, it will be automatically recognized by model file suffix, default is equivalent to OUTPUT_TYPE_AUTO

  # Hardware selection
  target_hardware: # Currently available AX620, AX630
  
  # Special handling of Tensor (old version), called tensor_conf, new version is recommended for more complex customization
  input_tensors      {}
  output_tensors     {}

  # Special handling of Tensor (new version)
  src_input_tensors {} # Attributes of the input tensor used to describe the input model, equivalent to input_tensors
  src_output_tensors {} # Attributes used to describe the output tensor of the input model
  dst_input_tensors {} # Attributes of the input tensor used to modify the output model, equivalent to output_tensors
  dst_output_tensors {} # Attributes used to modify the output tensor of the output model

  # cpu subgraph backend processing engine: ONNX OR AXE
  cpu_backend_settings {}

  # neuwizard parameters configuration
  neuwizard_conf { # Used to instruct Neuwizard to compile the onnx model into lava_joint format
    operator_conf {} # Used to formulate various cap operators
    dataset_conf_calibration {} # Used to describe the calibration dataset during compilation
  }

  # pulsar compiler configuration
  pulsar_conf {
    # pulsar_compiler is used to instruct pulsar_compiler to compile a lava_joint or lava format model into a joint or neu format model
    ...
  }

The ``config.prototxt`` needs to be properly configured according to the above structure.

.. attention::

  The ``input_tensors``, ``output_tensors`` options are kept for compatibility with older toolchains, while ``src_input_tensors`` and ``dst_input_tensors`` are equivalent to ``input_tensors`` and ``output_tensors``, and it is recommended to use the newer version -------------------------------------- Detailed description of the different modules of the configuration file -------------------------------------- This section details each ``sub_config`` in ``config.prototxt``.

.. _input_and_output_config:

~~~~~~~~~~~~~~~~~~~~~~
输入输出配置
~~~~~~~~~~~~~~~~~~~~~~

.. _input_path:

^^^^^^^^^^^^^^^^^^^^^^^
input_path
^^^^^^^^^^^^^^^^^^^^^^^

Property Description

.. list-table::
    :widths: 15 40
    :header-rows: 1

    - - Attributes
      - Description
    - - Parameter path
      - ``input_path``
    - - Parameter role
      - Specifies the path to the input model
    - - Parameter type
      - String
    - - Optional list
      - /
    - - Caution  
      - 1. The path is the relative path to the directory where the configuration file is located

        2. The parameter value string should be wrapped in double quotes ""

Code example

.. code-block:: sh
  :linenos:

  # input_path example
  input_path: "./model.onnx"

^^^^^^^^^^^^^^^^^^^^^^^
input_type
^^^^^^^^^^^^^^^^^^^^^^^

Property Description

.. list-table::
    :widths: 15 40
    :header-rows: 1

    - - Attributes
      - Description
    - - Parameter path
      - ``input_type``
    - - Parameter role
      - | Specify the type of the input model
        | By default, the compiler will automatically infer the model file name by its suffix. Sometimes the inferred result may not be what is expected
    - - Parameter type
      - Enum
    - - Optional list
      - ``INPUT_TYPE_ONNX``
    - - Caution  
      - Note that enum parameter values do not need to be quoted

Code example

.. code-block:: sh
  :linenos:

  # input_type example

  input_type: INPUT_TYPE_ONNX

.. _output_path:

^^^^^^^^^^^^^^^^^^^^^^^^
output_path
^^^^^^^^^^^^^^^^^^^^^^^^

Property Description

.. list-table::
    :widths: 15 40
    :header-rows: 1

    - - Attributes
      - Description
    - - Parameter path
      - ``output_path``
    - - Parameter role
      - Specifies the path to the output model
    - - Parameter type
      - String
    - - Optional list
      - /
    - - Notes  
      - Same as :ref:`input_path <input_path>`

Code example


.. code-block:: sh
  :linenos:

  # output_path example

  output_path: "./compiled.joint"

^^^^^^^^^^^^^^^^^^^^^^^^
output_type
^^^^^^^^^^^^^^^^^^^^^^^^

Property Description

.. list-table::
    :widths: 15 40
    :header-rows: 1

    - - Attributes
      - Description
    - - Parameter path
      - ``output_type``
    - - Parameter role
      - Specifies the type of the output model
    - - parameter type
      - Enum
    - - Optional list
      - ``OUTPUT_TYPE_JOINT``
    - - Caution  
      - Note that enum parameter values do not need to be quoted

Code example

.. code-block:: sh
  :linenos:

  # output_type example

  output_type: OUTPUT_TYPE_JOINT

.. _select_hardware:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
target_hardware
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Property Description

.. list-table::
    :widths: 15 40
    :header-rows: 1

    * - Attributes
      - Description
    * - parameter path
      - ``target_hardware``
    * - Parameter role
      - Specifies the hardware platform for which the compiled output model is to be used
    * - Parameter type
      - Enum
    * - Optional list
      - | ``TARGET_HARDWARE_AX630``
        | ``TARGET_HARDWARE_AX620``
    * - Caution
      - None


Code example

.. code-block:: sh
  :linenos:

  # target_hardware example

  target_hardware: TARGET_HARDWARE_AX630

.. tip::

  It is recommended to specify the hardware platform in the command line arguments to avoid model conversion errors due to the hardware platform.

.. _tensor_conf:

~~~~~~~~~~~~~~~~~~~~~~~~~~
tensor_conf
~~~~~~~~~~~~~~~~~~~~~~~~~~

^^^^^^^^^^^^^^^^^^^^^
Overview
^^^^^^^^^^^^^^^^^^^^^

.. Note::

  The ``Pulsar`` toolchain has the ability to adjust the properties of the input/output ``tensor`` of the output model, 
  i.e., allow the output model (e.g., ``joint`` model) to have input and output data properties (e.g., image size, color space, etc.) that do not match those of the original input model (e.g., ``onnx`` model).

**tensor_conf** configuration includes ``src_input_tensors`` , ``src_output_tensors`` , ``dst_input_tensors`` , ``dst_output_tensors`` . 

Property Description

.. list-table::
    :widths: 10 60
    :header-rows: 1

    - - Attributes
      - Description
    - - Parameter path
      - ``config_name`` itself, e.g. ``src_input_tensors``
    - - Parameter role
      - | ``src_input_tensors`` for the ``input tensor`` attribute of the ``description (description)`` input model
        | ``src_output_tensors`` for ``describing (description)`` the ``output tensor`` property of the input model
        | ``dst_input_tensors`` for ``modifying`` the ``input tensor`` properties of the output model
        | ``dst_output_tensors`` for ``modifying`` the output model's ``output tensor`` property
    - - parameter type
      - Struct
    - - optional list
      - /
    - - Cautions
      - None

^^^^^^^^^^^^^^^^^^^^^
Optional list
^^^^^^^^^^^^^^^^^^^^^

""""""""""""""""""""""
tensor_name
""""""""""""""""""""""

.. list-table::
    :widths: 10 60
    :header-rows: 1

    - - Properties
      - Description
    - - Parameter name
      - ``tensor_name``
    - - Parameter role
      - Specifies the name of the ``tensor`` of the input model described by the current structure or the ``tensor`` of the output model acted upon
    - - Parameter type
      - String
    - - Optional list
      - /
    - - Caution
      - For each of the arrays ``src_input_tensors`` , ``src_output_tensors`` , ``dst_input_tensors`` and ``dst_output_tensors`` , 
        If the ``tensor_name`` field in any of the ``item`` structures is default, then the contents of that ``item`` will overwrite the contents of the other ``item`` in the array

.. _color_space:

""""""""""""""""""""""
color_space
""""""""""""""""""""""

.. list-table::
    :widths: 10 60
    :header-rows: 1

    - - Properties
      - Description
    - - Parameter name
      - ``color_space``
    - - Parameter role
      - Used to describe the color space of the ``tensor`` of the input model, or to specify the color space of the ``tensor`` of the output model
    - - Parameter type
      - Enum
    - - Enum - optional list
      - | ``TENSOR_COLOR_SPACE_BGR``
        | ``TENSOR_COLOR_SPACE_RGB``
        | ``TENSOR_COLOR_SPACE_GRAY``
        | ``TENSOR_COLOR_SPACE_NV12``
        | ``TENSOR_COLOR_SPACE_NV21``
        | ``TENSOR_COLOR_SPACE_BGR0``
        | ``TENSOR_COLOR_SPACE_AUTO``
        | **DEFAULT:** ``TENSOR_COLOR_SPACE_AUTO`` , auto-identify based on model input channel number: 3-channel: BGR; 1-channel: GRAY
    - - Caution
      - None

.. _data_type:

""""""""""""""""""""""
data_type
""""""""""""""""""""""

.. list-table::
    :widths: 10 60
    :header-rows: 1

    - - Properties
      - Description
    - - Parameter name
      - ``data_type``
    - - Parameter role
      - Specifies the data type of the input and output ``tensor``.
    - - Parameter type
      - Enum
    - - Optional list
      - | ``DATA_TYPE_UNKNOWN``
        | ``UINT2``
        | ``INT2``
        | ``MINT2``
        | ``UINT4``
        | ``MINT4``
        | ``UINT8``
        | ``INT8``
        | ``MINT8``
        | ``UINT16``
        | ``INT16``
        | ``FLOAT32``
        | **DEFAULT:** ``UINT8`` is the default value for the input ``tensor`` , ```FLOAT32`` is the default value for the output ``tensor``
    - - Caution
      - None

.. _QValue:

""""""""""""""""""""""""""""""""""""""""""""
quantization_value
""""""""""""""""""""""""""""""""""""""""""""

An integer, often referred to as the ``Q`` value. It takes effect when configured as a positive number, or as a recommended value if one of the following conditions is met

  - The source model outputs real, the target model outputs integer
  - Source model input real, target model input integer

Code example

.. code-block:: sh

  # Configure Q values
  dst_output_tensors {
    data_type: INT16
    quantization_value: 256 # dynamic Q value when not configured
  }

.. hint::

  The ``Q`` value can be understood as a special ``affine`` operation. The ``Q`` value actually represents a ``scale`` , which can be converted to a specified fixed-point value field by dividing the output of the real number field by ``sclae``.
  into a specified fixed-point value field.

.. Note::

  There are two kinds of ``Q`` values:
    * Dynamic ``Q`` values are calculated dynamically from the maximum and minimum ranges in the ``calibration`` data set.
    * Static ``Q`` values are usually ``scale`` values that are manually specified by the user based on a priori information.

.. hint::

  The ``joint`` model contains information about the ``Q`` value, and the specific ``Q`` value is printed when ``run_joint`` is run.

.. attention::
  
  Using the ``Q`` value on the ``AX630`` saves a ``cpu affine`` operation, and therefore allows for speedup. The ``AX620`` supports ``float`` output, so even with the ``Q`` value, there is no speedup.

""""""""""""""""""""""
color_standard
""""""""""""""""""""""

.. list-table::
    :widths: 10 60
    :header-rows: 1

    - - Properties
      - Description
    - - Parameter name
      - ``color_standard``
    - - Parameter role
      - Used to set the color space standard
    - - Parameter type
      - Enum
    - - Optional list
      - | ``CSC_LEGACY``
        | ``CSS_ITU_BT601_STUDIO_SWING``
        | ``CSS_ITU_BT601_FULL_SWING``
        | **DEFAULT:** ``CSC_LEGACY``
    - - Caution
      - None

""""""""""""""""""""""
tensor_layout
""""""""""""""""""""""

.. list-table::
  :widths: 10 60
  :header-rows: 1

  - - Properties
    - Description
  - - Parameter name
    - ``tensor_layout``
  - - Parameter role
    - Used to modify the data layout
  - - Parameter type
    - Enum
  - - Optional list
    - | ``NHWC``
      | ``NCHW``
      | ``NATIVE`` Default, not recommended
  - - Notes
    - None

Code example

.. code-block:: sh
  :linenos:

  # target_hardware example

  src_input_tensors {
    color_space: TENSOR_COLOR_SPACE_AUTO
  }
  dst_output_tensors {
    color_space: TENSOR_COLOR_SPACE_NV12
  }

.. _cpu_backend_settings:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CPU subgraph settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

  ``AXEngine`` is ``AXera``'s own inference library, which can improve the ``FPS`` of the model to some extent, essentially replacing the ``CPU`` subgraph of ``ONNX`` with the ``AXE`` subgraph, and in terms of memory usage, the memory usage of the ``AXE`` subgraph on some models will be significantly reduced, and in the worst case, it will be the same as the original ``ONNX``.

.. list-table::
    :widths: 15 40
    :header-rows: 1

    * - Properties
      - Description
    * - Parameter path
      - ``cpu_backend_settings``
    * - Parameter role
      - Controls the ``CPU`` backend mode used by the compiled model, currently ``ONNX`` and ``AXEngine`` are available
    * - Parameter type
      - Struct
    * - Optional list
      - /
    * - Caution  
      - If you need to make a ``joint`` model with an ``AXEngine`` backend run on a ``BSP`` that does not support the ``AXEngine`` backend, you need to enable both ``onnx_setting.mode`` and ``axe_setting.mode`` for 

Code example

.. code-block:: sh
  :linenos:

  cpu_backend_settings {
    onnx_setting {
      mode: ENABLED
    }
    axe_setting {
      mode: ENABLED
      axe_param {
        optimize_slim_model: true
      }
    }
  }

Field Description

.. list-table::
    :header-rows: 1

    * - field name
      - Parameter path
      - Parameter Type
      - Parameter role
      - model
      - Notes
    * - ``onnx_setting``
      - cpu_backend_settings.onnx_setting
      - Struct
      - Controls whether the ``ONNX`` backend is enabled or not
      - DEFAULT / ENABLED / DISABLED, default is DEFAULT
      - DEFAULT and ENABLED are equivalent for ONNX
    * - ``axe_setting``
      - cpu_backend_settings.axe_setting
      - Struct
      - Controls whether the ``AXEngine`` backend is enabled or not
      - DEFAULT / ENABLED / DISABLED, default is DEFAULT
      - AXEngine's DEFAULT is equivalent to DISABLED
    * - ``optimize_slim_model``
      - cpu_backend_settings.axe_setting.axe_param.optimize_slim_model
      - Bool
      - Indicates whether optimization mode is enabled
      - No
      - Recommended when the network output feature map is small, otherwise not recommended

.. important::

  Users are recommended to use the ``CPU`` backend of ``AXE`` more often (the model ``initial`` is faster and better optimized for speed), the current ``ONNX`` backend support is for compatibility with older versions of the toolchain and will be deprecated in future releases.

.. _neuwizard_conf:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
neuwizard_conf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``neuwizard_conf`` contains a variety of configuration information, which can be configured to meet a variety of needs.

^^^^^^^^^^^^^^^^^^^^^^^^^^
operator_conf
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

  The ``operator_conf`` can be configured for input and output capping operations, where an additional capping operator appends an operation to the input or output ``tensor`` of an existing operator; in the configuration file, the process of adding a capping operator is done by expanding or modifying the properties of the input or output ``tensor`` of an existing operator.

Input-output capping operators enable pre-processing and post-processing of ``tensor``

.. list-table::
  :widths: 10 20 50
  :header-rows: 1

  - - Algorithm list
    - Type
    - Description
  - - ``input_conf_items``
    - Struct
    - Preprocessing operator, used to preprocess the input data for the model
  - - ``output_conf_items``
    - Struct
    - Post-processing operator, used to post-process the output data

Code examples

.. code-block::
  :name: gm_opr
  :linenos:

  # Example code, cannot be copied and used directly
  neuwizard_conf {
    operator_conf {
      input_conf_items {
        selector {
          ...
        selector { ... }
        attributes {
          # Array of preprocessing operators
          ...
        }
      }
      output_conf_items {
        selector {
          ...
        }
        attributes {
          # Array of post-processing operators
          ...
        }
      }
    }
  }

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Preprocessing and Preprocessing Operators
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Parameter paths

- ``neuwizard_conf.operator_conf.input_conf_items``

Example code

.. code-block:: sh
  :name: input_conf_items.pre
  :linenos:

  # Note that by parameter path, the following is placed in the appropriate location in the configuration file
  input_conf_items {
      # selector to indicate which input tensor the additional preprocessor operator will act on
      selector {
          op_name: "inp" # The name of the input tensor
      }
      # attributes to wrap the cap operator on "inp"
      attributes {
          input_modifications {
              # do an affine operation on the input data, which changes the input data type of the compiled model from floating point [0, 1) to uint8
              affine_preprocess {
                  slope: 1
                  slope_divisor: 255
                  bias: 0
                  }
          }
      }
  }

.. attention::

  ``affine`` is essentially a ``* k + b`` operation. 
  The ``affine`` operation in ``affine_preprocess`` is counter-intuitive, for example, changing the type of a floating-point field [0, 1) to UINT8 [0, 255] requires dividing by ``255`` instead of multiplying by ``255``, 
  while converting [0, 255] to floating point [0, 1] requires multiplying by ``255`` (configuring slope_divisor as ``0.00392156862745098``).

.. _input_conf_items_selector:

``input_conf_items.selector`` Property Description

.. list-table::
  :widths: 10 60
  :header-rows: 1

  - - Properties
    - Description
  - - parameter name
    - ``selector``
  - - Parameter path
    - :file:`neuwizard_conf.operator_conf.input_conf_items.selector`
  - - Parameter role
    - Name of the input tensor on which the additional preprocessing operator will act
  - - field description
    - | ``op_name`` specifies the full name of the input tensor. For example, "inp"
      | ``op_name_regex`` specifies a regular expression that will be used to adapt multiple tensors. The corresponding cap operator in the attributes structure will be applied to all adapted tensors

Code Example

.. code-block:: sh
  :name: input_conf_items.selector
  :linenos:

  # input_conf_items.selector 示例
  selector {
    op_name: "inp"
  }

.. _input_conf_items_attribute:

``input_conf_items.attributes`` 属性说明

.. list-table::
  :widths: 10 60
  :header-rows: 1

  * - Properties
    - Description
  * - parameter name
    - ``attributes``
  * - parameter path
    - :file:`neuwizard_conf.operator_conf.input_conf_items.attributes`
  * - parameter type
    - Struct
  * - Parameter role
    - Used to describe changes to the attributes of the input ``tensor``, the target input ``tensor`` is specified by ``input_conf_items.selector``
  * - field description
    - | ``type`` : Specifies or modifies the data type of the input ``tensor``. Enumeration type, default value ``DATA_TYPE_UNKNOWN``
      | ``input_modifications`` : Array of preprocessing operators, capping operators added to the input tensor. There are several of them, you can specify more than one at the same time

where ``type`` is an enumeration type, :ref:`click here <data_type>` to see the supported types. ``input_modifications`` is specified as follows:

.. list-table::
  :widths: 10 60
  :header-rows: 1

  * - Properties
    - Description
  * - field name
    - ``input_modifications``
  * - Type
    - Struct
  * - Function
    - Array of **preprocessing operators** that act on a particular input ``tensor``
  * - Caution
    - All operators in the array of preprocessing operators are executed sequentially, with the second operator in the array taking the output of the previous operator as input, and so on
    
**Preprocessing operator**

The preprocessing operators include ``input_normalization`` and ``affine_preprocess``.

.. list-table::
  :widths: 10 60
  :header-rows: 1
  :name: preprocessing operator [input_normalization]

  * - operator name
    - ``input_normalization``
  * - parameter path
    - neuwizard_conf.operator_conf.input_conf_items.attributes.input_modifications.input_normalization
  * - field descriptions
    - ``mean`` : an array of floating point numbers
      ``std`` : array of floating point numbers
  * - Effects
    - Implementation :math:`y = (x - mean) / std` .
  * - Caution:
    - | The order of ``mean/std`` is related to the :ref:`color space <color_space>` of the input ``tensor``.
      | If the above variables are equal to ``TENSOR_COLOR_SPACE_AUTO`` / ``TENSOR_COLOR_SPACE_BGR`` then the order of ``mean/std`` is ``BGR``.
      | If the above variables are equal to ``TENSOR_COLOR_SPACE_RGB`` then the order of ``mean/std`` is ``RGB``

.. _pre_affine_preprocess:

.. list-table::
  :widths: 10 60
  :header-rows: 1
  :name: preprocessing operator [affine_preprocess]

  * - the name of the operator
    - ``affine_preprocess``
  * - parameter path
    - neuwizard_conf.operator_conf.input_conf_items.attributes.input_modifications.affine_preprocess
  * - field descriptions
    - | ``slope`` : an array of floating point numbers
      | ``slope_divisor`` : an array of floating point numbers
      | ``bias`` : array of floating point numbers. The length of the array is the same as ``slope``.
      | ``bias_divisor`` : Array of floating point numbers. The length of the array is the same as ``slope``.
  * - Effects
    - Implementation :math:`y = x * (slope / slope\_divisor) + (bias / bias\_divisor)` .
  * - Caution:
    - None

Code example

.. code-block:: sh
  :name: input_conf_items.attributes.input_modifications.affine_preprocess
  :linenos:

  # Change the input data type from the number field {k / 255} (k=0, 1, ... , 255) to the integer field [0, 255], expecting the compiled model input data type to be uint8
  affine_preprocess {
    slope: 1
    slope_divisor: 255
    bias: 0
  }

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Postprocessing and Postprocessing Operators
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Parameter path

- ``neuwizard_conf.operator_conf.output_conf_items``

Code example

.. code-block:: sh
  :name: output_conf_items.post
  :linenos:

  # Note that by parameter path, the following is placed in the appropriate location in the configuration file
  output_conf_items {
      # selector to indicate the output tensor
      selector {
          op_name: "oup" # The name of the output tensor
      }
      # attributes for wrapping the cap operator on "oup"
      attributes {
          output_modifications {
              # do an affine operation on the output data to change the output data type of the compiled model from floating point [0, 1) to uint8
              affine_preprocess {
                  slope: 1
                  slope_divisor: 255
                  bias: 0 
                  }
          }
      }
  }

``output_conf_items.selector`` same as :ref:`input_conf_items.selector <input_conf_items_selector>` , ``output_conf_items.attributes`` same as :ref:`input_conf_items.attribute <input_conf_items_attribute>` .

**Postprocessing operator**

Postprocessing operator ``affine_preprocess``.

... list-table::
  :widths: 10 60
  :header-rows: 1
  :name: postprocessing operator [affine_preprocess]

  * - operator name
    - Operator description
  * - affine_preprocess
    - Do the ``affine`` operation on the output ``tensor``.

The rest is the same as :ref:`input_modifications.affine_preprocess <pre_affine_preprocess>`

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
dataset_conf_calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _calibration:

.. list-table::
  :widths: 10 60
  :header-rows: 1
  :name: dataset_conf_calibration

  * - operator name
    - ``dataset_conf_calibration``
  * - parameter path
    - neuwizard_conf.dataset_conf_calibration
  * - Role
    - To describe the dataset needed for the calibration process
  * - Caution:
    - The default ``batch_size`` is ``32``, if you get an ``Out Of Memory, OOM`` error, you can try to reduce the ``batch_size``.

Code example

.. code-block:: sh
  :name: output_conf_items
  :linenos:

  dataset_conf_calibration {
    path: "... /imagenet-1k-images.tar" # needs to be replaced with your own quantified data
    type: DATASET_TYPE_TAR # The type is tar
    size: 256 # An integer to represent the size of the dataset, which will be randomly sampled from the full set
    batch_size: 32 # An integer to represent the batch_size of the data used for internal parameter training, calibration or error detection during the model transfer process, default value is 32
  }

.. _pulsar_conf:

~~~~~~~~~~~~~~~~~~~~~~~~~~
pulsar_conf
~~~~~~~~~~~~~~~~~~~~~~~~~~

Property Description

... list-table::
    :widths: 15 40
    :header-rows: 1

    * - Attributes
      - Description
    * - parameter path
      - ``pulsar_conf``
    * - Parameter role
      - Configuration parameters for the compiler sub-tool ``pulsar_compiler``

        Used to instruct ``pulsar_compiler`` to compile ``lava_joint`` or ``lava`` format models into ``joint`` or ``neu`` format models
    * - parameter types
      - Struct
    * - optional list
      - /
    * - Caution  
      - Be careful to follow the path of the parameters to the correct location in the configuration file

Code examples

.. code-block:: sh
  :name: config.pulsar_conf
  :linenos:

  pulsar_conf {
    ax620_virtual_npu: AX620_VIRTUAL_NPU_MODE_111 # Compiled model uses virtual core #1 of ax620 virtual NPU 1+1 mode
    batch_size_option: BSO_DYNAMIC # The compiled model supports dynamic batch
    batch_size: 1
    batch_size: 2
    batch_size: 4 # Maximum batch_size is 4; requires high performance for inference with batch_size of 1 2 or 4
  }

Structure field descriptions

.. list-table::
    :header-rows: 1

    * - field name
      - Parameter Path
      - Parameter Type
      - Parameter role
      - Optional list
      - Notes
    * - ``virtual_npu``
      - pulsar_conf.virtual_npu
      - Enum
      - Specifies the ``AX630A`` virtual ``NPU`` core used by the target model
      - | ``VIRTUAL_NPU_MODE_AUTO``
        | ``VIRTUAL_NPU_MODE_0``
        | ``VIRTUAL_NPU_MODE_311``
        | ``VIRTUAL_NPU_MODE_312``
        | ``VIRTUAL_NPU_MODE_221``
        | ``VIRTUAL_NPU_MODE_222``
        | **DEFAULT:** ``VIRTUAL_NPU_MODE_AUTO``
      - | MODE_0 means no virtual NPU is used
        | This configuration item needs to be used if ``PulsarConfiguration.target_hardware`` is specified as ``TARGET_HARDWARE_AX630``.
        | This configuration item is used with ``ax620_virtual_npu``.
    * - ``ax620_virtual_npu``
      - pulsar_conf.ax620_virtual_npu
      - Enum
      - Specifies the ``AX620A`` virtual ``NPU`` core used by the target model
      - | ``AX620_VIRTUAL_NPU_MODE_AUTO``
        | ``AX620_VIRTUAL_NPU_MODE_0``
        | ``AX620_VIRTUAL_NPU_MODE_111``
        | ``AX620_VIRTUAL_NPU_MODE_112``
      - | MODE_0 means no virtual NPU is used
        | This configuration item needs to be used if ``PulsarConfiguration.target_hardware`` is specified as ``TARGET_HARDWARE_AX620``.
        | This configuration item is used with virtual_npu
    * - ``batch_size_option``
      - pulsar_conf.batch_size_option
      - Enum
      - Sets the ``batch`` type supported by the ``joint`` format model
      - | ``BSO_AUTO``
        | ``BSO_STATIC`` # Static ``batch``, fixed ``batch_size`` for inference, optimal performance
        | ``BSO_DYNAMIC`` # dynamic ``batch``, supports arbitrary ``batch_size`` up to the maximum value when inferring, more flexible to use
        | **DEFAULT:** ``BSO_AUTO`` , default is static ``batch``
      - None
    * - ``batch_size``
      - pulsar_conf.batch_size
      - IntArray
      - Sets the ``batch size`` supported by the ``joint`` format model, default is 1
      - /
      - | When ``batch_size_option`` is specified as ``BSO_STATIC``, ``batch_size`` indicates the unique ``batch size`` that the ``joint`` format model can use when reasoning
        | When ``batch_size_option`` is specified as ``BSO_DYNAMIC``, ``batch_size`` indicates the maximum ``batch size`` that can be used for ``joint`` format model inference.
        | When generating a ``joint`` format model that supports dynamic ``batch``, multiple values can be configured to improve performance when reasoning with ``batch size`` up to these values
        | Increase the size of ``joint`` format model files when multiple ``batch sizes`` are specified
        | ``batch_size_option`` will default to ``BSO_DYNAMIC`` when multiple ``batch_sizes`` are configured

.. _some_params_called_by_cmdline:

---------------------------------------------------------------
Parameters that can be passed via the command line
---------------------------------------------------------------

.. hint::

  Command line arguments override some of the corresponding configuration in the configuration file, and are only used to help with more complex functions that can be implemented through the configuration file.

.. list-table::
    :widths: 15 40
    :header-rows: 1

    - - Parameters
      - Description
    - - input
      - Input model path
    - - output
      - Output model path
    - - calibration_batch_size
      - The batch_size of the calibration dataset
    - - batch_size_option
      - {BSO_AUTO,BSO_STATIC,BSO_DYNAMIC}
    - - output_dir
      - Specify the output directory
    - - virtual_npu
      - Specify the virtual NPU
    - - input_tensor_color
      - {auto,rgb,bgr,gray,nv12,nv21}
    - - output_tensor_layout
      - {native,nchw,nhwc}
    - - color_std
      - {studio,full} only support nv12/nv21 now
    - - target_hardware 
      - {AX630,AX620,AX170} target hardware to compile
    - - enable_progress_bar
      - Whether to print progress bar, not enabled by default


----------------------------------------------
config.prototxt Minimal configuration
----------------------------------------------

Example of simplest_config.prototxt, which can be copied directly into a file and run.

.. code-block::
  :name: simplest_config.prototxt
  :linenos:

  # simplest_config.prototxt example, can be copied directly into the file and run
  input_type: INPUT_TYPE_ONNX # Specifies that the input model is of type onnx, if this field is omitted, the compiler will automatically infer the model file by its suffix, however, sometimes the inference may not be the desired result
  output_type: OUTPUT_TYPE_JOINT # Specifies that the output model is of type Joint
  src_input_tensors { # Attributes of the input tensor used to describe the input model
    color_space: TENSOR_COLOR_SPACE_AUTO # The compiler determines the color space itself
  }
  dst_input_tensors { # Attributes of the input tensor used to modify the output model
    color_space: TENSOR_COLOR_SPACE_AUTO # The compiler determines the color space itself
  }
  neuwizard_conf { # neuwizard parameter configuration
    operator_conf { # input_output_capping_configuration: additional input and output capping operators add an operation to the input or output tensor of an existing operator; in the configuration file, the process of adding a capping operator is done by expanding or modifying the properties of the input or output tensor of an existing operator
      input_conf_items { # used to preprocess the input data for the model
        attributes { # describe changes to the attributes of the input tensor, the target input tensor is specified by input_conf_items.selector, not specified, default is ? 
          input_modifications { # array of preprocessing operators, cap operators to be added to the input tensor, there are several, you can specify more than one, all operators in the preprocessing array are executed sequentially, the second operator in the array is the output of the previous operator as input, and so on
            affine_preprocess { # do an affine (i.e. x * k + b) operation on the input data, which changes the input data type of the compiled model from floating point [0, 1) to uint8
              slope: 1 # Array of floating point numbers. The length of the array is equal to 1 or the number of channels of the data. When the length is 1, the compiler will automatically copy the channel times
              slope_divisor: 255 # Array of floating point numbers. The length of the array is the same as slope
              bias: 0 # Array of floating point numbers. The length of the array is the same as slope
                                  # The effect is the same as: y = x * (slope / slope_divisor) + (bias / bias_divisor)
            }
          }
        }
      }
    }
    dataset_conf_calibration {
      path: ". /imagenet-1k-images.tar" # A tarball with 1000 images, used to calibrate the model during compilation
      type: DATASET_TYPE_TAR # The type is tar
      size: 256 # indicates the size of the dataset, which will be randomly sampled from the full set, batch_size defaults to 32
    batch_size defaults to 32.}
  }
  pulsar_conf { # pulsar compiler parameters configuration
    batch_size: 1 # Set the batch size supported by the joint format model, default is 1
  }

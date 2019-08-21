# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit.aqua.input import ClassificationInput
from qiskit.aqua import run_algorithm
from qiskit import Aer
from svm.datasets import ad_hoc_data, Wine

n = 2  # dimension of each data point

training_input, test_input, class_labels = ad_hoc_data(10, 10, n, 0.3)

datapoints, class_to_label = split_dataset_to_data_and_labels(test_input)

params = {
    'problem': {'name': 'classification', 'random_seed': 10598},
    'algorithm': {'name': 'VQC', 'override_SPSA_params': True},
    'backend': {'name': 'statevector_simulator'},
    'optimizer': {'name': 'SPSA', 'max_trials': 200, 'save_steps': 1},
    'variational_form': {'name': 'RYRZ', 'depth': 3},
    'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2}
}

algo_input = ClassificationInput(training_input, test_input, datapoints[0])

result = run_algorithm(params, algo_input)
# result = run_algorithm(params, algo_input, backend=Aer.get_backend('statevector_simulator'))
print(result)

# Copyright 2025 Google LLC
# Copyright 2025 Google LLC
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

set -x

run_unit_tests(){
    # Make sure you are inside brand-search-optimization directory
    # And ENABLE_UNIT_TEST_MODE=1 in .env
    export PYTHONPATH="$PYTHONPATH:."
    pytest tests/
}

run_unit_tests

exit 0
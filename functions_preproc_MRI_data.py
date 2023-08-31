# MIT License

# Copyright (c) 2023 Sarah H. Needleman

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# # # # # # 
# Sarah H. Needleman
# University College London

# Functions for pre-processing MRI data, including adding subjects
# to the masks dictionary for co-registration.

import json

def add_to_mask_dictionary(dictionary_name, SubjectID, Masks_list, Cpresent):
    """
    Function to add a subject and their mask names to the mask directory.
    Dictionary was created in mask_dictionary_script.py.

    Arguments:
    dictionary_name = the name of the dictionary (and dir) to add the entries to.
    SubjectID = name of the subject + V# + date.
    Masks_list = list of the masks reference image numbers, from which both the 
    cardiac masks and lung masks can be deduced.
    Cpresent = list of 0 and 1 for whether a cardiac mask is present for each slice
    in the slice ordering (A to P). 1 = Yes, 0 = No; list with four elements.
    Returns:
    None

    """
    # Mask 'endings' for lung and cardiac mask creation
    ending = ["-LungMask", "-CardiacMask"]

    # Read in original dictionary
    with open(dictionary_name) as json_file:
        data_read_in_json = json.load(json_file)

    # Add more dictionary entries
    # Add the entries using: dictionary_name[key] = value to add new value
    LungMasks_list = []
    CardiacMasks_list = []
    for j in range(len(Masks_list)):
        LungMasks_list.append(Masks_list[j] + ending[0])
        if Cpresent[j] ==1:
            CardiacMasks_list.append(Masks_list[j] + ending[1])
        else:
            CardiacMasks_list.append(Masks_list[j] + ending[0])

    # Add in key (new subject) with the values for that key (for that new subject)
    data_read_in_json[SubjectID] = {
        "LungMasks" : LungMasks_list,
        "CardiacMasks" : CardiacMasks_list
    }

    # Write this new dictionary entry to the original dictionary 
    # this will add a new entry/append the dictionary, rather than overwriting anything.
    json_data_read_in_json = json.dumps(data_read_in_json)
    with open(dictionary_name, 'w') as outfile:
        outfile.write(json_data_read_in_json)
    
    # No values to return.
    return




def load_mask_subj_names(dictionary_name, mask_type_lung, subject):
    """
    Function to load the mask names for a particular subject using the mask dictionary.

    Arguments:
    dictionary_name = name and location of the json dictionary file containing
    the subjects and their mask names.
    mask_type_lung = 1 if the mask is lung or 0 if the mask is thoracic.
    subject = subject ID (date, V#, initials) as in the directories and 
    in the dictionary.
    Returns:
    mask_names = list of mask names for a particular subject.

    """
    with open(dictionary_name) as json_file:
        data_read_in_json = json.load(json_file)
        # Access the values for the subject's key.
        # Use mask_type_lung to identify whether to access the LungMasks or
        # CardiacMasks.
        if mask_type_lung == 1:
            masks_load = 'LungMasks'
        else:
            # Must equal 0.
            masks_load = 'CardiacMasks'
        mask_names = data_read_in_json[subject][masks_load]
    return mask_names



if __name__ == "__main__":
    # ...
    a = 1
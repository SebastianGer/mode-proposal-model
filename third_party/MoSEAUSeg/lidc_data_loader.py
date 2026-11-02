# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)
# Source: https://github.com/gaozhitong/MoSE-AUSeg/blob/main/data/preprocess/lidc_data_loader.py
# Changes:
# - Added makefolder method from original utils.py file to lidc_data_loader.py, since we did not need any of the other code.
# - Changed the way the data is loaded from pickle file.
# - Added tqdm for progress tracking.
"""
                              Apache License
                        Version 2.0, January 2004
                     http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.

   "License" shall mean the terms and conditions for use, reproduction,
   and distribution as defined by Sections 1 through 9 of this document.

   "Licensor" shall mean the copyright owner or entity authorized by
   the copyright owner that is granting the License.

   "Legal Entity" shall mean the union of the acting entity and all
   other entities that control, are controlled by, or are under common
   control with that entity. For the purposes of this definition,
   "control" means (i) the power, direct or indirect, to cause the
   direction or management of such entity, whether by contract or
   otherwise, or (ii) ownership of fifty percent (50%) or more of the
   outstanding shares, or (iii) beneficial ownership of such entity.

   "You" (or "Your") shall mean an individual or Legal Entity
   exercising permissions granted by this License.

   "Source" form shall mean the preferred form for making modifications,
   including but not limited to software source code, documentation
   source, and configuration files.

   "Object" form shall mean any form resulting from mechanical
   transformation or translation of a Source form, including but
   not limited to compiled object code, generated documentation,
   and conversions to other media types.

   "Work" shall mean the work of authorship, whether in Source or
   Object form, made available under the License, as indicated by a
   copyright notice that is included in or attached to the work
   (an example is provided in the Appendix below).

   "Derivative Works" shall mean any work, whether in Source or Object
   form, that is based on (or derived from) the Work and for which the
   editorial revisions, annotations, elaborations, or other modifications
   represent, as a whole, an original work of authorship. For the purposes
   of this License, Derivative Works shall not include works that remain
   separable from, or merely link (or bind by name) to the interfaces of,
   the Work and Derivative Works thereof.

   "Contribution" shall mean any work of authorship, including
   the original version of the Work and any modifications or additions
   to that Work or Derivative Works thereof, that is intentionally
   submitted to Licensor for inclusion in the Work by the copyright owner
   or by an individual or Legal Entity authorized to submit on behalf of
   the copyright owner. For the purposes of this definition, "submitted"
   means any form of electronic, verbal, or written communication sent
   to the Licensor or its representatives, including but not limited to
   communication on electronic mailing lists, source code control systems,
   and issue tracking systems that are managed by, or on behalf of, the
   Licensor for the purpose of discussing and improving the Work, but
   excluding communication that is conspicuously marked or otherwise
   designated in writing by the copyright owner as "Not a Contribution."

   "Contributor" shall mean Licensor and any individual or Legal Entity
   on behalf of whom a Contribution has been received by Licensor and
   subsequently incorporated within the Work.

2. Grant of Copyright License. Subject to the terms and conditions of
   this License, each Contributor hereby grants to You a perpetual,
   worldwide, non-exclusive, no-charge, royalty-free, irrevocable
   copyright license to reproduce, prepare Derivative Works of,
   publicly display, publicly perform, sublicense, and distribute the
   Work and such Derivative Works in Source or Object form.

3. Grant of Patent License. Subject to the terms and conditions of
   this License, each Contributor hereby grants to You a perpetual,
   worldwide, non-exclusive, no-charge, royalty-free, irrevocable
   (except as stated in this section) patent license to make, have made,
   use, offer to sell, sell, import, and otherwise transfer the Work,
   where such license applies only to those patent claims licensable
   by such Contributor that are necessarily infringed by their
   Contribution(s) alone or by combination of their Contribution(s)
   with the Work to which such Contribution(s) was submitted. If You
   institute patent litigation against any entity (including a
   cross-claim or counterclaim in a lawsuit) alleging that the Work
   or a Contribution incorporated within the Work constitutes direct
   or contributory patent infringement, then any patent licenses
   granted to You under this License for that Work shall terminate
   as of the date such litigation is filed.

4. Redistribution. You may reproduce and distribute copies of the
   Work or Derivative Works thereof in any medium, with or without
   modifications, and in Source or Object form, provided that You
   meet the following conditions:

   (a) You must give any other recipients of the Work or
       Derivative Works a copy of this License; and

   (b) You must cause any modified files to carry prominent notices
       stating that You changed the files; and

   (c) You must retain, in the Source form of any Derivative Works
       that You distribute, all copyright, patent, trademark, and
       attribution notices from the Source form of the Work,
       excluding those notices that do not pertain to any part of
       the Derivative Works; and

   (d) If the Work includes a "NOTICE" text file as part of its
       distribution, then any Derivative Works that You distribute must
       include a readable copy of the attribution notices contained
       within such NOTICE file, excluding those notices that do not
       pertain to any part of the Derivative Works, in at least one
       of the following places: within a NOTICE text file distributed
       as part of the Derivative Works; within the Source form or
       documentation, if provided along with the Derivative Works; or,
       within a display generated by the Derivative Works, if and
       wherever such third-party notices normally appear. The contents
       of the NOTICE file are for informational purposes only and
       do not modify the License. You may add Your own attribution
       notices within Derivative Works that You distribute, alongside
       or as an addendum to the NOTICE text from the Work, provided
       that such additional attribution notices cannot be construed
       as modifying the License.

   You may add Your own copyright statement to Your modifications and
   may provide additional or different license terms and conditions
   for use, reproduction, or distribution of Your modifications, or
   for any such Derivative Works as a whole, provided Your use,
   reproduction, and distribution of the Work otherwise complies with
   the conditions stated in this License.

5. Submission of Contributions. Unless You explicitly state otherwise,
   any Contribution intentionally submitted for inclusion in the Work
   by You to the Licensor shall be under the terms and conditions of
   this License, without any additional terms or conditions.
   Notwithstanding the above, nothing herein shall supersede or modify
   the terms of any separate license agreement you may have executed
   with Licensor regarding such Contributions.

6. Trademarks. This License does not grant permission to use the trade
   names, trademarks, service marks, or product names of the Licensor,
   except as required for reasonable and customary use in describing the
   origin of the Work and reproducing the content of the NOTICE file.

7. Disclaimer of Warranty. Unless required by applicable law or
   agreed to in writing, Licensor provides the Work (and each
   Contributor provides its Contributions) on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
   implied, including, without limitation, any warranties or conditions
   of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
   PARTICULAR PURPOSE. You are solely responsible for determining the
   appropriateness of using or redistributing the Work and assume any
   risks associated with Your exercise of permissions under this License.

8. Limitation of Liability. In no event and under no legal theory,
   whether in tort (including negligence), contract, or otherwise,
   unless required by applicable law (such as deliberate and grossly
   negligent acts) or agreed to in writing, shall any Contributor be
   liable to You for damages, including any direct, indirect, special,
   incidental, or consequential damages of any character arising as a
   result of this License or out of the use or inability to use the
   Work (including but not limited to damages for loss of goodwill,
   work stoppage, computer failure or malfunction, or any and all
   other commercial damages or losses), even if such Contributor
   has been advised of the possibility of such damages.

9. Accepting Warranty or Additional Liability. While redistributing
   the Work or Derivative Works thereof, You may choose to offer,
   and charge a fee for, acceptance of support, warranty, indemnity,
   or other liability obligations and/or rights consistent with this
   License. However, in accepting such obligations, You may act only
   on Your own behalf and on Your sole responsibility, not on behalf
   of any other Contributor, and only if You agree to indemnify,
   defend, and hold each Contributor harmless for any liability
   incurred by, or claims asserted against, such Contributor by reason
   of your accepting any such warranty or additional liability.

END OF TERMS AND CONDITIONS

APPENDIX: How to apply the Apache License to your work.

   To apply the Apache License to your work, attach the following
   boilerplate notice, with the fields enclosed by brackets "[]"
   replaced with your own identifying information. (Don't include
   the brackets!)  The text should be enclosed in the appropriate
   comment syntax for the file format. We also recommend that a
   file or class name and description of purpose be included on the
   same "printed page" as the copyright notice for easier
   identification within third-party archives.

Copyright [yyyy] [name of copyright owner]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import pickle

import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def makefolder(folder):
    # Source: https://github.com/gaozhitong/MoSE-AUSeg/blob/main/utils/utils.py
    """
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False


def crop_or_pad_slice_to_size(slice, nx, ny):
    x, y = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s : x_s + nx, y_s : y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c : x_c + x, :] = slice[:, y_s : y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c : y_c + y] = slice[x_s : x_s + nx, :]
        else:
            slice_cropped[x_c : x_c + x, y_c : y_c + y] = slice[:, :]

    return slice_cropped


def find_subset_for_id(ids_dict, id):

    for tt in ["test", "train", "val"]:
        if id in ids_dict[tt]:
            return tt
    raise ValueError("id was not found in any of the train/test/val subsets.")


def prepare_data(input_file, output_file):
    """
    Main function that prepares a dataset from the raw challenge data to an hdf5 dataset
    """

    hdf5_file = h5py.File(output_file, "w")
    # max_bytes = 2 ** 31 - 1

    # data = {}
    file_path = os.fsdecode(input_file)
    # bytes_in = bytearray(0)
    # input_size = os.path.getsize(file_path)
    # with open(file_path, 'rb') as f_in:
    #    for i in range(0, input_size, max_bytes):
    #        bytes_in += f_in.read(max_bytes)
    # new_data = pickle.loads(bytes_in)
    # data.update(new_data)
    data = pickle.load(open(file_path, "rb"))
    series_uid = []

    for key, value in tqdm(data.items(), desc="Reading series uids."):
        series_uid.append(value["series_uid"])

    unique_subjects = np.unique(series_uid)

    split_ids = {}
    # Add random_state to ensure same splits.
    train_and_val_ids, split_ids["test"] = train_test_split(
        unique_subjects, test_size=0.2, random_state=0
    )
    split_ids["train"], split_ids["val"] = train_test_split(
        train_and_val_ids, test_size=0.2, random_state=0
    )

    images = {}
    labels = {}
    uids = {}
    groups = {}

    for tt in ["train", "test", "val"]:
        images[tt] = []
        labels[tt] = []
        uids[tt] = []
        groups[tt] = hdf5_file.create_group(tt)

    for key, value in tqdm(data.items(), desc="Reading in data."):

        s_id = value["series_uid"]

        tt = find_subset_for_id(split_ids, s_id)

        images[tt].append(value["image"].astype(float) - 0.5)

        lbl = np.asarray(value["masks"])  # this will be of shape 4 x 128 x 128
        lbl = lbl.transpose((1, 2, 0))

        labels[tt].append(lbl)
        uids[tt].append(hash(s_id))  # Checked manually that there are no collisions

    for tt in tqdm(["test", "train", "val"], desc="Creating data subsets."):

        groups[tt].create_dataset("uids", data=np.asarray(uids[tt], dtype=np.int))
        groups[tt].create_dataset("labels", data=np.asarray(labels[tt], dtype=np.uint8))
        groups[tt].create_dataset("images", data=np.asarray(images[tt], dtype=np.float))

    # hdf5_file.close()


def load_and_maybe_process_data(
    input_file, preprocessing_folder, force_overwrite=False
):
    """
    This function is used to load and if necessary preprocesses the LIDC challenge data

    :param input_folder: Folder where the raw ACDC challenge data is located
    :param preprocessing_folder: Folder where the proprocessed data should be written to
    :param force_overwrite: Set this to True if you want to overwrite already preprocessed data [default: False]

    :return: Returns an h5py.File handle to the dataset
    """

    data_file_name = "data_lidc.hdf5"

    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        prepare_data(input_file, data_file_path)
    else:
        pass

    return h5py.File(data_file_path, "r")


def save_npy(data, vis_dir):
    stages = ["train", "val", "test"]
    types = ["images", "labels"]
    for s in tqdm(stages, desc="saving npy files: train/val/test"):
        for t in types:
            if not os.path.exists(os.path.join(vis_dir, s, t)):
                os.makedirs(os.path.join(vis_dir, s, t))
            for index, image in tqdm(
                enumerate(data[s][t]), desc="saving npy files: images/labels"
            ):
                np.save(os.path.join(vis_dir, s, t, str(index) + ".npy"), image)
    return


# def vis(data, vis_dir):
#     import matplotlib.pyplot as plt
#     stages = [  'test']
#     types = ['images', 'labels']
#     for s in stages:
#         if not os.path.exists(os.path.join(vis_dir, s)):
#             os.makedirs(os.path.join(vis_dir, s))
#         # for t in types:
#         L = len(data[s]['images'])
#         index = 0
#         while index < L:
#             image_list = data[s]['images']
#             label_list = data[s]['labels']
#             i = 0
#             N = 5
#             plt.figure(figsize = (5*3,N*3))
#             for image_id in range(index, index + N):
#                 image = image_list[image_id]
#                 labels = label_list[image_id]
#                 i += 1
#                 plt.subplot(N,5,i)
#                 plt.imshow(image)
#                 plt.axis('off')
#                 for label_idx in range(labels.shape[-1]):
#                     # import ipdb; ipdb.set_trace()
#                     label = labels[:,:,label_idx]
#                     i += 1
#                     plt.subplot(N,5,i)
#                     plt.imshow(label)
#                     plt.axis('off')
#             plt.savefig(os.path.join(vis_dir, s, str(image_id) + '.png'))
#             index = image_id
#
#
#     return

if __name__ == "__main__":
    data_root = "my_path/data_lidc.pickle"
    preproc_folder = "my_path/lidc/preproc"
    npy_dir = "my_path/lidc/npy"

    save_npy(
        load_and_maybe_process_data(data_root, preproc_folder, force_overwrite=True),
        npy_dir,
    )

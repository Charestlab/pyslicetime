
from dcmstack.extract import default_extractor
import pydicom
import os

data_path = '/media/charesti-start/data/irsa-fmri/dicom/CBU101295/20100930_101706/Series_003_CBU_EPI_BOLD_216/'

dicom_file = os.path.join(
    data_path, '1.3.12.2.1107.5.2.32.35119.2010093010310825996437574.dcm')

ds = pydicom.dcmread(dicom_file)

meta = default_extractor(
    ds)

st_times = np.asarray(meta['CsaImage.MosaicRefAcqTimes'])/1000

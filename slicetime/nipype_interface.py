from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec
from nipype.utils.filemanip import split_filename
from slicetime.main import run_slicetime
import os


class SliceTimeInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        desc='volume to be slice-time interpolated',
        mandatory=True)

    out_file = File(
        name_template='%s_tshift',
        desc='output image file name',
        name_source='in_file')

    tr_old = traits.Float(desc='what is the acquisition TR',
                          mandatory=True)

    tr_new = traits.Float(desc='what is the new TR for interpolation',
                          mandatory=True)

    sliceorder = traits.ListInt(desc='what is the sliceorder for slice-time',
                                mandatory=True)


class SliceTimeOutputSpec(TraitedSpec):
    slicetimed_volume = File(desc="slice-time interpolated volume")


class SliceTime(BaseInterface):
    input_spec = SliceTimeInputSpec
    output_spec = SliceTimeOutputSpec

    def _run_interface(self, runtime):

        run_slicetime(
            inpath=self.inputs.in_file,
            outpath=self.inputs.out_file,
            sliceorder=self.inputs.sliceorder,
            tr_old=self.inputs.tr_old,
            tr_new=self.inputs.tr_new,
        )

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["slicetimed_volume"] = self.inputs.out_file
        return outputs

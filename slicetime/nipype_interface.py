from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec, isdefined
from nipype.utils.filemanip import fname_presuffix
from slicetime.main import run_slicetime
import os


class SliceTimeInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        desc='volume to be slice-time interpolated',
        mandatory=True
    )

    out_file = File(
        desc='output image file name',
    )

    tr_old = traits.Float(
        desc='what is the acquisition TR',
        mandatory=True
    )

    tr_new = traits.Float(
        desc='what is the new TR for interpolation',
        mandatory=True
    )

    slicetimes = traits.ListFloat(
        desc='what are the slicetimes',
        mandatory=True
    )


class SliceTimeOutputSpec(TraitedSpec):
    out_file = File(
        desc="slice-time interpolated volume"
    )


class SliceTime(BaseInterface):
    input_spec = SliceTimeInputSpec
    output_spec = SliceTimeOutputSpec

    def _run_interface(self, runtime):

        run_slicetime(
            inpath=self.inputs.in_file,
            outpath=self._gen_outfilename(),
            slicetimes=self.inputs.slicetimes,
            tr_old=self.inputs.tr_old,
            tr_new=self.inputs.tr_new,
        )

        return runtime

    def _gen_outfilename(self):
        out_file = self.inputs.out_file
        if not isdefined(out_file) and isdefined(self.inputs.in_file):
            out_file = fname_presuffix(self.inputs.in_file, suffix='_stc')
        return os.path.abspath(out_file)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._gen_outfilename()
        return outputs

import torch
from torch.distributed.algorithms.join import (
    Join,
    Joinable,
    JoinHook,
)
from torch.distributed.utils import (
    _verify_param_shape_across_processes,
    _sync_module_states,
    _to_kwargs,
)
from torch.nn.parallel.distributed import _find_tensors, _tree_flatten_with_rref, _DDPSink, _tree_unflatten_with_rref

def ddp_forward(self, *inputs, **kwargs):
    with torch.autograd.profiler.record_function(
        "DistributedDataParallel.forward"
    ):
        if torch.is_grad_enabled() and self.require_backward_grad_sync:
            assert self.logger is not None
            self.logger.set_runtime_stats_and_log()
            self.num_iterations += 1
            self.reducer.prepare_for_forward()

        work = Join.notify_join_context(self)
        if work:
            self.reducer._set_forward_pass_work_handle(
                work, self._divide_by_initial_world_size  # type: ignore[arg-type]
            )

        if torch.is_grad_enabled() and self.reducer._rebuild_buckets():
            self.logger.info(
                "Reducer buckets have been rebuilt in this iteration."
            )
            self._has_rebuilt_buckets = True

        if self._check_sync_bufs_pre_fwd():
            self._sync_buffers()

        if self._join_config.enable:
            self._check_global_requires_backward_grad_sync(
                is_joined_rank=False
            )
        module_to_run = (
            self._replicated_tensor_module
            if self._use_replicated_tensor_module
            else self.module
        )

        if self.device_ids:
            inputs, kwargs = _to_kwargs(
                inputs,
                kwargs,
                self.device_ids[0],
                self.use_side_stream_for_tensor_copies,
            )
            with self._inside_ddp_forward():
                output = module_to_run.train_step(*inputs[0], **kwargs[0])  # type: ignore[index]
        else:
            with self._inside_ddp_forward():
                output = module_to_run.train_step(*inputs, **kwargs)

        if self._check_sync_bufs_post_fwd():
            self._sync_buffers()

        if torch.is_grad_enabled() and self.require_backward_grad_sync:
            self.require_forward_param_sync = True
            if self.find_unused_parameters and not self.static_graph:
                self.reducer.prepare_for_backward(
                    list(_find_tensors(output))
                )
            else:
                self.reducer.prepare_for_backward([])
        else:
            self.require_forward_param_sync = False

    if (self.find_unused_parameters and not self.static_graph) or (
        self.static_graph and self.num_iterations == 1
    ):
        state_dict = {
            "static_graph": self.static_graph,
            "num_iterations": self.num_iterations,
        }

        (
            output_tensor_list,
            treespec,
            output_is_rref,
        ) = _tree_flatten_with_rref(output)
        output_placeholders = [None for _ in range(len(output_tensor_list))]
        for i, output in enumerate(output_tensor_list):
            if torch.is_tensor(output) and output.grad_fn is None:
                output_placeholders[i] = output

        passthrough_tensor_list = _DDPSink.apply(
            self.reducer,
            state_dict,
            *output_tensor_list,
        )
        for i in range(len(output_placeholders)):
            if output_placeholders[i] is None:
                output_placeholders[i] = passthrough_tensor_list[i]

        output = _tree_unflatten_with_rref(
            output_placeholders, treespec, output_is_rref
        )
    return output
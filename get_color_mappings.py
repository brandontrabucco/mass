from mass.thor.segmentation_config import SegmentationConfig
from collections import OrderedDict


if __name__ == "__main__":

    # create task arguments for either the training and testing tasks
    task_sampler_params = SegmentationConfig\
        .stagewise_task_sampler_args(stage="train", process_ind=0,
                                     total_processes=1, devices=[0])

    # ensure object_id_to_color is populated by setting the
    # controller argument renderObjectImage = True
    task_sampler_params["ground_truth"] = True
    task_sampler_params["thor_controller_kwargs"] = \
        dict(renderInstanceSegmentation=True)

    # generate a sampler for training or testing evaluation
    task_sampler = SegmentationConfig.make_sampler_fn(
        **task_sampler_params, force_cache_reset=True,
        only_one_unshuffle_per_walkthrough=True, epochs=1)

    pickable_map = OrderedDict()
    openable_map = OrderedDict()
    pushable_map = OrderedDict()

    for task_id in range(task_sampler.length // 2):

        # skip the initial walkthrough phase of each training task
        task = task_sampler.next_task()
        task.step(action=task.action_names().index('done'))

        # set the unshuffle phase to the done state for scene evaluation
        task = task_sampler.next_task()
        task.step(action=task.action_names().index('done'))

        # get the poses of all objects in the scene
        unshuffle_poses, walkthrough_poses, _ = task.env.poses

        for object_one, object_two in zip(
                unshuffle_poses, walkthrough_poses):

            # for each object determine if is is misplaced
            if not task.env.are_poses_equal(object_one, object_two):

                color = task.env.controller\
                    .last_event.object_id_to_color[object_one["type"]]

                if object_two["pickupable"]:
                    pickable_map[object_one["type"]] = color

                elif object_two["openness"] is not None:
                    openable_map[object_one["type"]] = color

                else:  # catch any other objects
                    pushable_map[object_one["type"]] = color

        print()
        print()
        print(task_id)
        print()
        print("pickable_map", pickable_map)
        print()
        print("openable_map", openable_map)
        print()
        print("pushable_map", pushable_map)

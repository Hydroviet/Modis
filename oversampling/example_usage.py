    image_generator = SimpleImageGenerator(
        crop_size=args.crop_size,
        random_crop=args.random_crop,
        rotation_range=args.rotation_range,
        rotation_offset=args.rotation_offset,
        translate_range=args.translate_range,
        zoom_range=args.zoom_range,
        isotropic_zoom=args.isotropic_zoom,
        horizontal_flip=args.horizontal_flip,
        vertical_flip=args.vertical_flip,
        contrast_range=args.contrast_range,
        blurring_radius=args.blurring_radius,
        fillval=127.5,
        preprocessing_function=keras.imagenet_utils.preprocess_input)

    train_data_iterator = image_generator.flow_from_list(
        train_images,
        y=train_labels,
        is_training=True,
        batch_size=args.batch_size,
        seed=args.seed,
        balancing=None)

    net.train(
        train_data_iterator,
        val_data_iterator,
        learning_policy=args.learning_policy,
        base_learning_rate=args.base_learning_rate*10,
        learning_rate_decay_factor=args.learning_rate_decay_factor,
        learning_rate_decay_step=args.learning_rate_decay_step,
        learning_rate_power=args.learning_rate_power,
        momentum=args.momentum,
        epochs=5,
        batch_size=warmup_batch_size,
        checkpoints_dir=args.checkpoints_dir,
        log_dir=args.log_dir)
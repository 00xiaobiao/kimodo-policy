def create_action_provider(env,args):
    """create action provider based on parameters"""
    if args.action_source == "twist2_wholebody":
        from action_provider.action_provider_wh_twist2 import TWIST2ActionProvider
        return TWIST2ActionProvider(
            env=env,
            args_cli=args
        )
    elif args.action_source == "sonic_wholebody":
        from action_provider.action_provider_sonic import SonicActionProvider
        return SonicActionProvider(env=env, args_cli=args)
    elif args.action_source == "replay":
        if args.gmt_backend == "twist2":
            from action_provider.action_provider_wh_twist2 import TWIST2ActionProvider
            return TWIST2ActionProvider(env=env, args_cli=args)
        elif args.gmt_backend == "sonic":
            from action_provider.action_provider_sonic import SonicActionProvider
            return SonicActionProvider(env=env, args_cli=args)
        from action_provider.action_provider_replay import FileActionProviderReplay
        return FileActionProviderReplay(env=env,args_cli=args)
    else:
        print(f"unknown action source: {args.action_source}")
        return None

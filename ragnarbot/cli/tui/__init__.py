"""TUI onboarding wizard for ragnarbot."""

from rich.console import Console

from ragnarbot.cli.tui.components import QuitOnboarding, clear_screen
from ragnarbot.cli.tui.screens import (
    provider_screen,
    auth_method_screen,
    token_input_screen,
    model_screen,
    telegram_screen,
    summary_screen,
)
from ragnarbot.config.providers import PROVIDERS, get_models, get_provider, supports_oauth


def run_onboarding(console: Console) -> None:
    """Run the full onboarding wizard."""
    try:
        _onboarding_loop(console)
    except QuitOnboarding:
        clear_screen(console)
        console.print("\n  Setup cancelled.\n")
    except KeyboardInterrupt:
        clear_screen(console)
        console.print("\n  Setup cancelled.\n")


def _onboarding_loop(console: Console) -> None:
    """Main onboarding state machine with back navigation."""
    # State
    provider_idx: int | None = None
    auth_idx: int | None = None
    token: str | None = None
    model_idx: int | None = None
    telegram_token: str | None = None

    step = 1  # 1=provider, 2=auth, 3=token, 4=model, 5=telegram, 6=summary

    while True:
        if step == 1:
            provider_idx = provider_screen(console)
            if provider_idx is None:
                # Quit from first screen
                raise QuitOnboarding()
            step = 2

        elif step == 2:
            provider_id = PROVIDERS[provider_idx]["id"]
            if supports_oauth(provider_id):
                auth_idx = auth_method_screen(console, provider_id)
                if auth_idx is None:
                    step = 1
                    continue
            else:
                auth_idx = 1  # api_key
            step = 3

        elif step == 3:
            provider_id = PROVIDERS[provider_idx]["id"]
            auth_method = "oauth" if auth_idx == 0 else "api_key"
            token = token_input_screen(console, provider_id, auth_method)
            if token is None:
                # Go back to auth or provider
                if supports_oauth(provider_id):
                    step = 2
                else:
                    step = 1
                continue
            step = 4

        elif step == 4:
            provider_id = PROVIDERS[provider_idx]["id"]
            model_idx = model_screen(console, provider_id)
            if model_idx is None:
                step = 3
                continue
            step = 5

        elif step == 5:
            telegram_token = telegram_screen(console)
            if telegram_token is None:
                step = 4
                continue
            step = 6

        elif step == 6:
            provider_id = PROVIDERS[provider_idx]["id"]
            provider = get_provider(provider_id)
            auth_method = "oauth" if auth_idx == 0 else "api_key"
            models = get_models(provider_id)
            model = models[model_idx]
            telegram_configured = bool(telegram_token)

            ok = summary_screen(
                console,
                provider["name"],
                auth_method,
                model["name"],
                telegram_configured,
            )
            if not ok:
                step = 5
                continue

            # Save everything
            _save_results(
                console=console,
                provider_id=provider_id,
                auth_method=auth_method,
                token=token,
                model_id=model["id"],
                telegram_token=telegram_token if telegram_configured else "",
            )
            return


def _save_results(
    console: Console,
    provider_id: str,
    auth_method: str,
    token: str,
    model_id: str,
    telegram_token: str,
) -> None:
    """Save onboarding results to config and credentials files."""
    from ragnarbot.config.loader import load_config, save_config, get_config_path
    from ragnarbot.auth.credentials import (
        load_credentials,
        save_credentials,
        get_credentials_path,
    )
    from ragnarbot.utils.helpers import get_workspace_path

    # Load existing or create new
    config = load_config()
    creds = load_credentials()

    # Update config
    config.agents.defaults.model = model_id
    config.agents.defaults.auth_method = auth_method

    # Update telegram
    if telegram_token:
        config.channels.telegram.enabled = True

    save_config(config)

    # Update credentials (targeted — only touch the selected provider)
    provider_creds = getattr(creds.providers, provider_id)
    if auth_method == "oauth":
        provider_creds.oauth_key = token
    else:
        provider_creds.api_key = token

    if telegram_token:
        creds.channels.telegram.bot_token = telegram_token

    save_credentials(creds)

    # Ensure workspace exists
    workspace = get_workspace_path()

    # Create workspace templates if needed
    from ragnarbot.cli.commands import _create_workspace_templates
    _create_workspace_templates(workspace)

    clear_screen(console)
    console.print()
    console.print("  [green]Configuration saved![/green]")
    console.print()
    console.print(f"  Config:      {get_config_path()}")
    console.print(f"  Credentials: {get_credentials_path()}")
    console.print(f"  Workspace:   {workspace}")
    console.print()
    console.print("  [bold]Next steps:[/bold]")
    console.print("  Chat: [cyan]ragnarbot agent -m \"Hello!\"[/cyan]")
    if telegram_token:
        console.print("  Start gateway: [cyan]ragnarbot gateway[/cyan]")
    console.print()

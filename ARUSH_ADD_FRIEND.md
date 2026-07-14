# Arush — add friend (no implementation)

Checklist only. Do **not** start PriorityKV research coding in this step.

## 1. GitHub

1. Open https://github.com/Arush777/Priority_KV/settings/access  
   (or **Settings → Collaborators**)
2. **Add people** → invite friend’s GitHub username  
3. Permission: **Write**
4. Tell them to accept the invite email/notification

## 2. Telegram

1. Open group **Information_Retrieval agents**
2. Add your friend as a member
3. Confirm bot `@arush_ir_collab_bot` is still in the group
4. **Privately** (DM, not GitHub, not public channel) send:

```text
TELEGRAM_BOT_TOKEN=<your bot token>
TELEGRAM_CHAT_ID=-5470510083
Bot: @arush_ir_collab_bot
Group: Information_Retrieval agents
Repo: https://github.com/Arush777/Priority_KV
Doc: FRIEND_AGENT_SETUP.md on main
```

## 3. Send them this message

```text
Priority_KV collab setup (bridge only — no research coding yet)

1) Accept GitHub invite: https://github.com/Arush777/Priority_KV
2) Join Telegram group: Information_Retrieval agents
3) Clone:
   git clone git@github.com:Arush777/Priority_KV.git
   cd Priority_KV
4) Follow FRIEND_AGENT_SETUP.md exactly (AGENT_ID=friend, CURSOR_MODEL=auto)
5) I will send bot token + chat id in a private message
6) When `python -m collab_bridge ping` works, paste the prompt from
   FRIEND_AGENT_SETUP.md into your Cursor agent
7) Do not implement the plan until we both say setup is done
```

## 4. After they ping

In Telegram you should see `[agent:friend] bridge online`.  
Then reply here in Cursor: **friend online** — we’ll sync daemons only (still no plan implementation unless you ask).

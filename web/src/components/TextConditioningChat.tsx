import * as Dialog from "@radix-ui/react-dialog";
import { FormEvent, useEffect, useState } from "react";

type Message = { role: "operator" | "system"; text: string };

export function TextConditioningChat({
  instruction,
  defaultInstruction,
  scene,
  disabled,
  onInstructionChange,
}: {
  instruction: string;
  defaultInstruction: string;
  scene: "pot" | "drawer";
  disabled: boolean;
  onInstructionChange: (instruction: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  useEffect(() => { if (open) setDraft(instruction); }, [open, instruction]);
  useEffect(() => {
    setDraft("");
    setMessages([]);
  }, [scene]);
  const send = (event: FormEvent) => {
    event.preventDefault();
    const next = draft.trim();
    if (!next || disabled) return;
    onInstructionChange(next);
    setMessages((current) => [...current,
      { role: "operator", text: next },
      { role: "system", text: "Current instruction updated. Press Generate to run it." },
    ]);
    setDraft(next);
  };
  const reset = () => {
    onInstructionChange(defaultInstruction);
    setMessages((current) => [...current, { role: "system", text: "Restored the scene instruction." }]);
  };
  return <Dialog.Root open={open} onOpenChange={setOpen}>
    <Dialog.Trigger asChild><button type="button" className="button button-secondary" disabled={disabled}>Text chat</button></Dialog.Trigger>
    <Dialog.Portal>
      <Dialog.Overlay className="dialog-overlay" />
      <Dialog.Content className="freeplay-dialog text-chat-dialog">
        <header className="dialog-header"><div><Dialog.Title className="text-balance">Text conditioning</Dialog.Title><Dialog.Description>Edit the current instruction. Generate runs the selected setup.</Dialog.Description></div><Dialog.Close asChild><button type="button" className="button button-secondary">Close chat</button></Dialog.Close></header>
        <div className="text-chat-body">
          <div className="text-chat-messages" aria-live="polite">
            <p className="chat-message chat-system">Current condition: {instruction}</p>
            {messages.map((message, index) => <p key={index} className={`chat-message chat-${message.role}`}>{message.text}</p>)}
          </div>
          <form className="text-chat-form" onSubmit={send}>
            <label htmlFor="cosmos-text-condition">Current instruction</label>
            <textarea id="cosmos-text-condition" value={draft} disabled={disabled} maxLength={1000} rows={3} placeholder={defaultInstruction} onChange={(event) => setDraft(event.target.value)} />
            <div className="text-chat-actions"><button type="submit" className="button button-primary" disabled={disabled || !draft.trim()}>Update instruction</button><button type="button" className="button button-secondary" disabled={disabled || instruction === defaultInstruction} onClick={reset}>Restore tested instruction</button></div>
          </form>
          <p className="judge-muted">The VLM evaluates the generated video against your current instruction.</p>
        </div>
      </Dialog.Content>
    </Dialog.Portal>
  </Dialog.Root>;
}

import React from "react";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import { Scale, ThumbsUp, ThumbsDown, Copy, ShieldCheck, Sparkles, FileText } from "lucide-react";
import { motion } from "framer-motion";
import { FlowAnimation } from "./FlowAnimation";
import { DocumentScanner } from "./DocumentScanner";

const AIMessageActions = ({ content }: { content: string }) => {
  const [copied, setCopied] = React.useState(false);
  const [feedback, setFeedback] = React.useState<'up' | 'down' | null>(null);

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex items-center gap-2 mt-3 opacity-0 group-hover:opacity-100 transition-opacity" data-html2canvas-ignore="true">
      <button 
        onClick={handleCopy}
        className="p-1.5 rounded bg-[var(--color-bg-subtle)] text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-bg-surface)] transition-colors border border-[var(--color-border-dim)]" title="Copy Message">
        <Copy className={`w-3.5 h-3.5 ${copied ? 'text-[var(--color-accent)]' : ''}`} />
      </button>
      <button 
        onClick={() => setFeedback('up')}
        className={`p-1.5 rounded bg-[var(--color-bg-subtle)] hover:bg-[var(--color-bg-surface)] transition-colors border border-[var(--color-border-dim)] ${feedback === 'up' ? 'text-[var(--color-accent)]' : 'text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)]'}`} title="Helpful">
        <ThumbsUp className={`w-3.5 h-3.5 ${feedback === 'up' ? 'fill-current' : ''}`} />
      </button>
      <button 
        onClick={() => setFeedback('down')}
        className={`p-1.5 rounded bg-[var(--color-bg-subtle)] hover:bg-[var(--color-bg-surface)] transition-colors border border-[var(--color-border-dim)] ${feedback === 'down' ? 'text-red-500' : 'text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)]'}`} title="Not Helpful">
        <ThumbsDown className={`w-3.5 h-3.5 ${feedback === 'down' ? 'fill-current' : ''}`} />
      </button>
    </div>
  );
};

export interface Message {
  role: "user" | "ai";
  content: string;
  intent?: string;
  timestamp?: Date;
  referenced_nodes?: { name: string, type: string, description: string }[];
}

interface ChatMessageListProps {
  messages: Message[];
  loading: boolean;
  setInput: (v: string) => void;
  textareaRef: React.RefObject<HTMLTextAreaElement | null>;
  messagesEndRef: React.RefObject<HTMLDivElement | null>;
  suggestedPrompts: string[];
  exportRef: React.RefObject<HTMLDivElement | null>;
}

export function ChatMessageList({
  messages,
  loading,
  setInput,
  textareaRef,
  messagesEndRef,
  suggestedPrompts,
  exportRef,
}: ChatMessageListProps) {
  return (
    <div className="flex-1 overflow-y-auto pt-8 pb-32 px-4 scrollbar-thin bg-[var(--color-bg-base)] relative">
      {/* Ambient background glow */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none z-0 fixed">
        <motion.div 
          className="absolute top-1/4 left-1/4 w-[50%] h-[50%] rounded-full bg-[var(--color-accent-glass)] blur-[120px]"
          animate={{ x: [0, 30, 0], y: [0, 20, 0], scale: [1, 1.05, 1] }}
          transition={{ duration: 20, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div 
          className="absolute bottom-1/4 right-1/4 w-[40%] h-[40%] rounded-full bg-blue-600/10 blur-[100px]"
          animate={{ x: [0, -40, 0], y: [0, -30, 0], scale: [1, 1.1, 1] }}
          transition={{ duration: 15, repeat: Infinity, ease: "easeInOut", delay: 2 }}
        />
      </div>

      <div ref={exportRef} className="w-full min-h-full flex flex-col p-4 relative z-10">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center max-w-[640px] mx-auto pb-12">
            <motion.div 
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.8, type: "spring" }}
              className="relative w-20 h-20 flex items-center justify-center mb-8"
            >
              <div className="absolute inset-0 bg-yellow-500/20 blur-2xl rounded-full"></div>
              <Scale className="w-16 h-16 stroke-[1.5] text-yellow-400 drop-shadow-[0_0_15px_rgba(250,204,21,0.5)]" />
            </motion.div>
            <motion.h2 
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="font-heading font-semibold text-3xl md:text-4xl text-[var(--color-text-primary)] text-center mb-10 tracking-tight"
            >
              What legal matter can I help with?
            </motion.h2>
            
            <motion.div 
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.4 }}
              className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full" data-html2canvas-ignore="true"
            >
              {suggestedPrompts.map((prompt, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setInput(prompt);
                    textareaRef.current?.focus();
                  }}
                  className="group relative flex flex-col items-start bg-[var(--color-glass)] backdrop-blur-md border border-[var(--color-glass-border)] rounded-xl p-5 text-left transition-all hover:bg-[var(--color-glass-hover)] hover:border-[var(--color-border-accent)] hover:shadow-[0_0_20px_var(--color-accent-glow)] overflow-hidden"
                >
                  <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-30 group-hover:scale-110 transition-all text-[var(--color-accent)]">
                    <Sparkles className="w-8 h-8" />
                  </div>
                  <div className="w-8 h-8 rounded-lg bg-[var(--color-accent-glass)] border border-[var(--color-accent-glass)] flex items-center justify-center mb-4 text-[var(--color-accent)] group-hover:scale-110 transition-transform">
                    <FileText className="w-4 h-4" />
                  </div>
                  <span className="text-[15px] font-medium text-[var(--color-text-primary)] leading-snug relative z-10">{prompt}</span>
                </button>
              ))}
            </motion.div>
          </div>
        ) : (
          <div className="max-w-[720px] mx-auto w-full flex flex-col gap-8">
            {messages.map((msg, i) => (
              <div key={i} className={`flex w-full group ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                
                {msg.role === "user" ? (
                  <div className="flex items-end gap-2 max-w-[70%]">
                    <span className="text-[10px] text-[var(--color-text-muted)] opacity-0 group-hover:opacity-100 transition-opacity mb-1 shrink-0">
                      {msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) : ''}
                    </span>
                    <div className="bg-gradient-to-br from-emerald-500 to-teal-600 text-[var(--color-text-primary)] px-4 py-3 rounded-2xl rounded-tr-sm text-[15px] font-sans font-normal leading-relaxed whitespace-pre-wrap shadow-[0_5px_15px_rgba(16,185,129,0.2)]">
                      {msg.content}
                    </div>
                  </div>
                ) : (
                  <div className="flex items-start gap-3 w-full pl-4 border-l-2 border-[var(--color-border-accent)]">
                    <div className="flex-1 min-w-0">
                      
                      <div className="flex items-center gap-2 mb-2">
                        {msg.intent && (
                          <span className="inline-flex items-center px-2 py-1 rounded bg-[var(--color-accent-muted)] border border-[var(--color-border-accent)] border-opacity-30 text-[10px] font-sans font-medium text-[var(--color-accent)] tracking-wide uppercase">
                            {msg.intent}
                          </span>
                        )}
                        <span className="text-[10px] text-[var(--color-text-muted)] opacity-0 group-hover:opacity-100 transition-opacity">
                          {msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) : ''}
                        </span>
                      </div>
                      
                      <div className="text-[15px] font-sans leading-[1.8] text-[var(--color-text-primary)]">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          rehypePlugins={[rehypeRaw]}
                          components={{
                            // eslint-disable-next-line @typescript-eslint/no-explicit-any
                            a: ({node, ...props}: any) => <a className="text-[var(--color-semantic-blue)] hover:underline hover:opacity-80 transition-all font-medium" target="_blank" rel="noopener noreferrer" {...props} />,
                            // eslint-disable-next-line @typescript-eslint/no-explicit-any
                            p: ({node, ...props}: any) => <motion.p initial={{opacity: 0, y: 10}} animate={{opacity: 1, y: 0}} transition={{duration: 0.4}} className="mb-4 last:mb-0" {...props} />,
                            // eslint-disable-next-line @typescript-eslint/no-explicit-any
                            h1: ({node, ...props}: any) => <motion.h1 initial={{opacity: 0, y: 10}} animate={{opacity: 1, y: 0}} transition={{duration: 0.4}} className="font-heading font-medium text-xl mt-6 mb-3" {...props} />,
                            // eslint-disable-next-line @typescript-eslint/no-explicit-any
                            h2: ({node, ...props}: any) => <motion.h2 initial={{opacity: 0, y: 10}} animate={{opacity: 1, y: 0}} transition={{duration: 0.4}} className="font-heading font-medium text-lg mt-5 mb-2" {...props} />,
                            // eslint-disable-next-line @typescript-eslint/no-explicit-any
                            h3: ({node, ...props}: any) => <motion.h3 initial={{opacity: 0, y: 10}} animate={{opacity: 1, y: 0}} transition={{duration: 0.4}} className="font-heading font-medium text-[16px] mt-4 mb-2" {...props} />,
                            // eslint-disable-next-line @typescript-eslint/no-explicit-any
                            ul: ({node, ...props}: any) => <motion.ul initial={{opacity: 0}} animate={{opacity: 1}} transition={{duration: 0.5}} className="list-disc pl-5 mb-4" {...props} />,
                            // eslint-disable-next-line @typescript-eslint/no-explicit-any
                            ol: ({node, ...props}: any) => <motion.ol initial={{opacity: 0}} animate={{opacity: 1}} transition={{duration: 0.5}} className="list-decimal pl-5 mb-4" {...props} />,
                            // eslint-disable-next-line @typescript-eslint/no-explicit-any
                            li: ({node, ...props}: any) => <motion.li initial={{opacity: 0, x: -5}} animate={{opacity: 1, x: 0}} transition={{duration: 0.3}} className="mb-1" {...props} />,
                            // eslint-disable-next-line @typescript-eslint/no-explicit-any
                            code: ({node, inline, ...props}: any) => 
                              inline 
                                ? <code className="font-mono text-[13px] bg-[var(--color-bg-subtle)] px-1.5 py-0.5 rounded text-[var(--color-text-primary)]" {...props} />
                                : <motion.code initial={{opacity: 0}} animate={{opacity: 1}} className="block font-mono text-[13px] bg-[var(--color-bg-subtle)] p-4 rounded border border-[var(--color-border-dim)] overflow-x-auto my-4 whitespace-pre-wrap" {...props} />,
                          }}
                        >
                          {msg.content}
                        </ReactMarkdown>
                      </div>
                      
                      {/* Action Buttons for AI Message */}
                      <AIMessageActions content={msg.content} />
                      
                      {/* Verified Badge (only if not loading and it's a draft or advice) */}
                      {!loading && (msg.intent?.includes("Draft") || msg.intent?.includes("Advice") || msg.intent?.includes("RTI") || msg.intent?.includes("Notice")) && (
                        <motion.div 
                          initial={{ scale: 1.5, opacity: 0 }}
                          animate={{ scale: 1, opacity: 1 }}
                          transition={{ type: "spring", stiffness: 200, damping: 15, delay: 0.5 }}
                          className="mt-4 inline-flex items-center gap-1.5 px-3 py-1.5 bg-[var(--color-accent-glass)] border border-[var(--color-accent-glass)] rounded-full text-[var(--color-accent)] text-[11px] font-semibold tracking-wide uppercase shadow-sm"
                        >
                          <ShieldCheck className="w-3.5 h-3.5" />
                          Reflexion Verified
                        </motion.div>
                      )}

                      {msg.referenced_nodes && msg.referenced_nodes.length > 0 && (
                        <div className="mt-5 flex flex-wrap gap-2" data-html2canvas-ignore="true">
                          {msg.referenced_nodes.map((node, idx) => (
                            <motion.div 
                              key={idx} 
                              initial={{ opacity: 0, scale: 0.8, y: 10 }}
                              animate={{ opacity: 1, scale: 1, y: 0 }}
                              transition={{ delay: 0.1 * idx, type: "spring", stiffness: 100 }}
                              className="group/chip relative flex items-center gap-2 px-3 py-1.5 bg-[var(--color-bg-subtle)] border border-[var(--color-border-dim)] hover:border-[var(--color-accent)] hover:shadow-[0_0_8px_rgba(var(--color-accent-rgb),0.15)] rounded-full transition-all cursor-default"
                            >
                              <div className="w-1.5 h-1.5 rounded-full bg-[var(--color-accent)] animate-pulse"></div>
                              <span className="text-[11px] font-medium text-[var(--color-text-primary)]">{node.name}</span>
                              
                              {/* Tooltip on hover */}
                              {node.description && (
                                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 p-2 bg-[#1a1a1a] border border-[#333] rounded-md shadow-xl opacity-0 invisible group-hover/chip:opacity-100 group-hover/chip:visible transition-all text-[10px] text-[var(--color-text-primary)]/80 z-10 pointer-events-none">
                                  {node.description}
                                </div>
                              )}
                            </motion.div>
                          ))}
                        </div>
                      )}

                    </div>
                  </div>
                )}
              </div>
            ))}

            {/* LOADING INDICATOR */}
            {loading && (
              <div className="flex items-start gap-3 w-full" data-html2canvas-ignore="true">
                {messages.length > 0 && messages[messages.length - 1].content.includes("Uploaded document:") ? (
                  <DocumentScanner />
                ) : (
                  <FlowAnimation />
                )}
              </div>
            )}
            
            <div ref={messagesEndRef} className="h-[150px] shrink-0" data-html2canvas-ignore="true" />
          </div>
        )}
      </div>
    </div>
  );
}

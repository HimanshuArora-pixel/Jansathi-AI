"use client";

import Link from "next/link";
import { ArrowRight, Sparkles, Scale, Shield, FileText, CheckCircle2 } from "lucide-react";
import { ThemeToggle } from "@/components/ThemeToggle";
import { ArchitectureShowcase } from "@/components/ArchitectureShowcase";
import { motion, useScroll, useTransform } from "framer-motion";
import { useRef } from "react";

const STAGGER_CHILD_VARIANTS: any = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: "easeOut" } },
};

export default function Home() {
  const containerRef = useRef(null);
  const { scrollY } = useScroll();
  const navBgOpacity = useTransform(scrollY, [0, 50], [0, 0.85]);
  const navBorderOpacity = useTransform(scrollY, [0, 50], [0, 0.4]);
  const navBackdropBlur = useTransform(scrollY, [0, 50], ["blur(0px)", "blur(12px)"]);

  return (
    <div ref={containerRef} className="min-h-screen bg-[var(--color-bg-base)] text-[var(--color-text-primary)] selection:bg-[var(--color-accent)] selection:text-[#080808] overflow-hidden relative">
      
      {/* FLUID ANIMATED BACKGROUND */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none z-0">
        <motion.div 
          className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] rounded-full bg-[var(--color-accent-muted)] blur-[120px]"
          animate={{ x: [0, 50, 0], y: [0, 30, 0], scale: [1, 1.1, 1] }}
          transition={{ duration: 15, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div 
          className="absolute top-[20%] right-[-10%] w-[60%] h-[60%] rounded-full bg-blue-600/15 blur-[150px]"
          animate={{ x: [0, -70, 0], y: [0, -40, 0], scale: [1, 1.2, 1] }}
          transition={{ duration: 18, repeat: Infinity, ease: "easeInOut", delay: 2 }}
        />
        <div className="absolute inset-0 bg-[url('/noise.png')] opacity-[0.03] mix-blend-overlay"></div>
      </div>

      {/* NAVBAR */}
      <motion.nav 
        style={{ 
          backgroundColor: useTransform(navBgOpacity, v => `rgba(5, 5, 5, ${v})`),
          borderBottom: useTransform(navBorderOpacity, v => `1px solid rgba(16, 185, 129, ${v})`),
          backdropFilter: navBackdropBlur,
          WebkitBackdropFilter: navBackdropBlur
        }}
        className="fixed top-0 left-0 right-0 h-[70px] z-50 transition-colors"
      >
        <div className="max-w-7xl mx-auto px-6 h-full flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2 font-heading font-semibold text-xl tracking-tight">
            <span className="text-[var(--color-text-primary)]">Jan</span>
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-blue-500">Saathi</span>
          </Link>
          <div className="flex items-center gap-6">
            <ThemeToggle />
            <Link 
              href="/login" 
              className="hidden sm:block text-sm font-medium text-[var(--color-text-primary)]/80 hover:text-[var(--color-text-primary)] transition-colors"
            >
              Sign in
            </Link>
            <Link 
              href="/chat" 
              className="group relative h-[38px] px-5 flex items-center justify-center bg-transparent overflow-hidden rounded-full font-medium text-sm transition-all hover:scale-105"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-emerald-500 to-blue-600 transition-transform duration-300 group-hover:scale-110"></div>
              <span className="relative text-[var(--color-text-primary)] flex items-center gap-2">
                Start for Free
              </span>
            </Link>
          </div>
        </div>
      </motion.nav>

      {/* HERO SECTION */}
      <section className="relative min-h-screen pt-[70px] pb-20 flex flex-col justify-center items-center z-10">
        <div className="max-w-7xl mx-auto px-6 w-full grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-8 items-center">
          
          {/* Left Text Content */}
          <motion.div 
            className="lg:col-span-7 flex flex-col items-start pt-12 lg:pt-0"
            variants={{
              hidden: { opacity: 0 },
              show: {
                opacity: 1,
                transition: { staggerChildren: 0.15 }
              }
            }}
            initial="hidden"
            animate="show"
          >
            {/* Eyebrow badge */}
            <motion.div variants={STAGGER_CHILD_VARIANTS} className="inline-flex items-center gap-2 px-3 py-1.5 mb-8 rounded-full bg-[var(--color-bg-surface)] border border-[var(--color-accent-glass)] shadow-[0_0_15px_var(--color-accent-glow)]">
              <Sparkles className="w-3.5 h-3.5 text-[var(--color-accent)]" />
              <span className="text-[11px] font-medium text-[var(--color-accent)] uppercase tracking-widest">
                India's First Agentic Legal AI
              </span>
            </motion.div>
            
            <motion.h1 variants={STAGGER_CHILD_VARIANTS} className="font-heading font-semibold text-[46px] sm:text-[56px] lg:text-[68px] text-[var(--color-text-primary)] leading-[1.05] tracking-[-0.03em] mb-6">
              Navigating Indian Law,<br/>
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 via-teal-300 to-blue-500">
                AI Legal Assistance
              </span><br/>
              For a Brighter Future.
            </motion.h1>
            
            <motion.p variants={STAGGER_CHILD_VARIANTS} className="text-[17px] sm:text-[19px] text-[var(--color-text-secondary)] leading-[1.6] max-w-[500px] mb-10 font-light">
              Empower yourself with 24/7 AI-driven legal support, expert guidance, document drafting, and complex cases simplified in seconds, all within your reach.
            </motion.p>
            
            <motion.div variants={STAGGER_CHILD_VARIANTS} className="flex flex-col sm:flex-row items-center gap-4 w-full sm:w-auto">
              <Link 
                href="/chat"
                className="group w-full sm:w-auto px-8 py-4 bg-[var(--color-accent)] hover:bg-[var(--color-accent-dim)] text-white font-semibold rounded-full flex items-center justify-center gap-2 transition-all shadow-[0_0_20px_var(--color-border-accent)] hover:shadow-[0_0_30px_rgba(16,185,129,0.5)] hover:scale-105"
              >
                Start for Free <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
              </Link>
              <Link 
                href="#features"
                className="w-full sm:w-auto px-8 py-4 bg-transparent border border-slate-700 text-[var(--color-text-primary)] hover:bg-[var(--color-glass-hover)] font-medium rounded-full text-center transition-colors"
              >
                Learn More
              </Link>
            </motion.div>
          </motion.div>

          {/* Right side 3D Glassmorphic Document */}
          <div className="lg:col-span-5 relative w-full h-[500px] hidden lg:flex items-center justify-center perspective-[1200px]">
            <motion.div
              className="relative w-[340px] h-[480px] preserve-3d"
              animate={{ 
                rotateY: [-5, 5, -5],
                rotateX: [5, -5, 5],
                y: [-10, 10, -10]
              }}
              transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
            >
              {/* Back Layer (Shadow/Glow) */}
              <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/20 to-blue-500/20 rounded-2xl blur-[40px] transform translate-z-[-20px]"></div>
              
              {/* Main Glass Card */}
              <div className="absolute inset-0 bg-[var(--color-glass)] backdrop-blur-xl border border-[var(--color-glass-border)] rounded-2xl shadow-2xl p-6 overflow-hidden flex flex-col">
                
                {/* Header */}
                <div className="flex justify-between items-start mb-6 border-b border-[var(--color-glass-border)] pb-4">
                  <div>
                    <h3 className="text-[var(--color-text-primary)] font-semibold text-lg tracking-wide">LEGAL DOCUMENT</h3>
                    <p className="text-[var(--color-text-secondary)] text-xs">Sub: Contract Agreement</p>
                  </div>
                  <span className="text-[10px] font-mono text-[var(--color-accent)] bg-emerald-400/10 px-2 py-1 rounded">JanSaathi</span>
                </div>

                {/* Content Skeleton */}
                <div className="flex-1 space-y-5">
                  <div className="space-y-2">
                    <div className="h-2 w-full bg-[var(--color-bg-elevated)] rounded"></div>
                    <div className="h-2 w-5/6 bg-[var(--color-bg-elevated)] rounded"></div>
                    <div className="h-2 w-4/5 bg-[var(--color-bg-elevated)] rounded"></div>
                  </div>
                  
                  <div>
                    <h4 className="text-[var(--color-text-primary)]/80 text-xs font-semibold mb-2">Section 1: Obligations</h4>
                    <div className="space-y-2">
                      <div className="h-2 w-full bg-[var(--color-bg-elevated)] rounded"></div>
                      <div className="h-2 w-11/12 bg-[var(--color-bg-elevated)] rounded"></div>
                      <div className="h-2 w-full bg-[var(--color-bg-elevated)] rounded"></div>
                    </div>
                  </div>

                  <div>
                    <h4 className="text-[var(--color-text-primary)]/80 text-xs font-semibold mb-2">Section 2: Liabilities</h4>
                    <div className="space-y-2">
                      <div className="h-2 w-[90%] bg-[var(--color-bg-elevated)] rounded"></div>
                      <div className="h-2 w-[85%] bg-[var(--color-bg-elevated)] rounded"></div>
                      <div className="h-2 w-[60%] bg-[var(--color-bg-elevated)] rounded"></div>
                    </div>
                  </div>
                </div>

                {/* Action Button at bottom */}
                <div className="mt-4 flex justify-end">
                  <div className="px-4 py-1.5 rounded-full bg-[var(--color-accent-muted)] border border-[var(--color-accent-glass)] text-[10px] font-semibold text-[var(--color-accent)] flex items-center gap-1.5">
                    <CheckCircle2 className="w-3 h-3" />
                    Generated by AI
                  </div>
                </div>
                
                {/* Glossy Overlay */}
                <div className="absolute inset-0 bg-gradient-to-tr from-white/0 via-white/[0.05] to-white/0 transform translate-z-[1px] pointer-events-none rounded-2xl"></div>
              </div>
            </motion.div>
          </div>
          
        </div>
      </section>

      {/* SOCIAL PROOF BAR */}
      <section className="relative z-10 w-full bg-[var(--color-bg-base)]/80 backdrop-blur-md border-y border-[var(--color-border-dim)] py-10">
        <div className="max-w-7xl mx-auto px-6 grid grid-cols-2 md:grid-cols-4 gap-8 text-center divide-x divide-[var(--color-border-dim)]">
          <div>
            <div className="font-heading font-bold text-4xl text-[var(--color-text-primary)] mb-2">10k+</div>
            <div className="text-xs font-medium text-[var(--color-text-secondary)] tracking-wider uppercase">Users Assisted</div>
          </div>
          <div>
            <div className="font-heading font-bold text-4xl text-[var(--color-text-primary)] mb-2">1,420</div>
            <div className="text-xs font-medium text-[var(--color-text-secondary)] tracking-wider uppercase">RTIs Drafted</div>
          </div>
          <div>
            <div className="font-heading font-bold text-4xl text-[var(--color-text-primary)] mb-2">94.8%</div>
            <div className="text-xs font-medium text-[var(--color-text-secondary)] tracking-wider uppercase">AI Accuracy</div>
          </div>
          <div>
            <div className="font-heading font-bold text-4xl text-[var(--color-text-primary)] mb-2">24/7</div>
            <div className="text-xs font-medium text-[var(--color-text-secondary)] tracking-wider uppercase">Instant Support</div>
          </div>
        </div>
      </section>

      {/* FEATURES SECTION */}
      <div className="relative z-10 bg-[var(--color-bg-base)]">
        <ArchitectureShowcase />
      </div>

      {/* FOOTER */}
      <footer className="relative z-10 w-full border-t border-[var(--color-border-dim)] py-12 bg-[var(--color-bg-base)]">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <span className="font-heading font-semibold text-xl text-[var(--color-text-primary)] tracking-tight">Jan<span className="text-[var(--color-accent)]">Saathi</span></span>
            <span className="text-[var(--color-text-primary)]/20">|</span>
            <span className="text-sm text-[var(--color-text-muted)]">Empowering Indian citizens</span>
          </div>
          <div className="flex items-center gap-8 text-sm font-medium text-[var(--color-text-secondary)]">
            <Link href="#" className="hover:text-[var(--color-accent)] transition-colors">Privacy Policy</Link>
            <Link href="#" className="hover:text-[var(--color-accent)] transition-colors">Terms of Service</Link>
            <Link href="/login" className="hover:text-[var(--color-accent)] transition-colors">Sign In</Link>
          </div>
        </div>
      </footer>

    </div>
  );
}

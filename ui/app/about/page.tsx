export default function AboutPage() {
  return (
    <div className="h-full flex items-center justify-center overflow-auto pb-40">
      <div className="max-w-2xl px-6 py-12 text-center">
        <div className="flex items-center justify-center mb-8">
          <img src="/logo-full.svg" alt="Telescope" className="h-12" />
        </div>

        <div className="space-y-4 text-base text-foreground/60 leading-relaxed text-justify">
          <p>
            Telescope is a framework to post-train LLMs with reinforcement
            learning for reasoning and agents.
          </p>
          <p>
            It comes with a trainer repository to run the training process
            easily, with a highly configurable setup that supports both simple
            and complex environments, tested on clusters with hundreds of GPUs.
          </p>
          <p>
            This visualization tool provides critical insight into your runs:
            metrics, rollouts, infrastructure, inference traces, trainer GPU
            events, and more.
          </p>
          <p>
            Telescope is built with a modern stack and focused on
            infrastructure, observability, scalability, and reliability.
          </p>
          <p>
            You can find the documentation at{" "}
            <a
              href="https://docs.telescope.training"
              target="_blank"
              rel="noopener noreferrer"
              className="underline underline-offset-2 hover:text-foreground transition-colors"
            >
              docs.telescope.training
            </a>
            .
          </p>
        </div>

        <div className="mt-10 flex items-center justify-center gap-6 text-xs text-muted-foreground">
          <a
            href="https://github.com/eduardoslonski/telescope"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-foreground transition-colors"
          >
            Trainer
          </a>
          <span className="text-border">|</span>
          <a
            href="https://github.com/eduardoslonski/telescope-ui"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-foreground transition-colors"
          >
            UI Visualization
          </a>
          <span className="text-border">|</span>
          <a
            href="https://eduardoslonski.com"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-foreground transition-colors"
          >
            Created by Eduardo Slonski
          </a>
        </div>
      </div>
    </div>
  )
}

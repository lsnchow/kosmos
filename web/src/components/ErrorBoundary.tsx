import { Component, type ErrorInfo, type ReactNode } from "react";

type Props = {
  children: ReactNode;
  /** Names the region, so a failed panel does not read as a failed product. */
  region?: string;
  onReset?: () => void;
};

type State = { error?: Error };

/**
 * A render throw used to blank the entire page. Mid-pitch that is
 * indistinguishable from the product not existing.
 *
 * The boundary reports what failed and where, keeps the rest of the console
 * mounted when it wraps a region, and offers a retry. It deliberately does not
 * substitute placeholder content for the failed region.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = {};

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // eslint-disable-next-line no-console
    console.error(`PLUMB render error in ${this.props.region ?? "console"}`, error, info.componentStack);
  }

  private reset = () => {
    this.setState({ error: undefined });
    this.props.onReset?.();
  };

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    const region = this.props.region ?? "console";
    return (
      <div className="boundary-fallback" role="alert">
        <div>
          <strong>{region} could not render</strong>
          <p className="text-pretty">
            This panel threw while rendering, so its contents are unavailable rather than
            approximated. Every other panel is unaffected.
          </p>
          <code>{error.message}</code>
        </div>
        <button type="button" className="button button-secondary" onClick={this.reset}>
          Retry this panel
        </button>
      </div>
    );
  }
}

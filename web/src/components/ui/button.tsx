import { forwardRef, type ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

const VARIANTS = {
  default: "border border-input bg-accent text-foreground hover:bg-border",
  primary: "border border-primary bg-primary text-primary-foreground hover:bg-primary-hover",
  ghost: "border border-transparent bg-transparent text-muted-foreground hover:bg-accent hover:text-foreground",
  danger: "border border-transparent bg-transparent text-destructive hover:bg-accent",
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: keyof typeof VARIANTS;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", ...props }, ref) => (
    <button
      ref={ref}
      className={cn(
        "inline-flex cursor-pointer items-center justify-center gap-1.5 rounded px-3 py-1.5 text-[13px] focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none disabled:cursor-default disabled:opacity-50",
        VARIANTS[variant],
        className,
      )}
      {...props}
    />
  ),
);
Button.displayName = "Button";

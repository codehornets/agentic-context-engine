"""
ACE-specific tracing decorators for Opik integration.

Provides convenient decorators for tracing ACE framework components
including roles (Generator, Reflector, Curator) and adaptation processes.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Dict, Optional, TypeVar, Union

from .opik_integration import get_integration

logger = logging.getLogger(__name__)

# Type variable for decorated functions
F = TypeVar('F', bound=Callable[..., Any])


def ace_track(
    name: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Callable[[F], F]:
    """
    General-purpose ACE tracing decorator.

    Args:
        name: Custom name for the trace (defaults to function name)
        tags: Additional tags for the trace
        metadata: Additional metadata for the trace
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            integration = get_integration()
            if not integration.is_available():
                return func(*args, **kwargs)

            trace_name = name or f"ace_{func.__name__}"
            trace_tags = (tags or []) + ["ace-framework"]

            try:
                # Import here to avoid circular imports
                from .opik_integration import track, opik_context

                @track(name=trace_name, tags=trace_tags, metadata=metadata)
                def traced_func():
                    return func(*args, **kwargs)

                return traced_func()
            except Exception as e:
                logger.warning(f"Failed to trace {func.__name__}: {e}")
                return func(*args, **kwargs)

        return wrapper
    return decorator


def track_role(
    role_name: Optional[str] = None,
    track_performance: bool = True
) -> Callable[[F], F]:
    """
    Decorator for ACE role methods (Generator, Reflector, Curator).

    Args:
        role_name: Name of the role (auto-detected if not provided)
        track_performance: Whether to track execution time and success
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            integration = get_integration()
            if not integration.is_available():
                return func(*args, **kwargs)

            # Auto-detect role name from class or use provided name
            detected_role = role_name
            if not detected_role and args:
                instance = args[0]
                detected_role = instance.__class__.__name__
            detected_role = detected_role or func.__name__

            start_time = time.time()
            success = False
            result = None

            try:
                # Import here to avoid circular imports
                from .opik_integration import track, opik_context

                @track(
                    name=f"role_{detected_role.lower()}_{func.__name__}",
                    tags=["ace-framework", "role", f"role-{detected_role.lower()}"]
                )
                def traced_role():
                    nonlocal success, result
                    result = func(*args, **kwargs)
                    success = True
                    return result

                traced_result = traced_role()

                # Log performance metrics if enabled
                if track_performance:
                    execution_time = time.time() - start_time
                    integration.log_role_performance(
                        role_name=detected_role,
                        execution_time=execution_time,
                        success=success,
                        input_data={"args_count": len(args), "kwargs_keys": list(kwargs.keys())},
                        output_data={"result_type": type(traced_result).__name__}
                    )

                return traced_result

            except Exception as e:
                execution_time = time.time() - start_time

                # Log failed performance
                if track_performance:
                    integration.log_role_performance(
                        role_name=detected_role,
                        execution_time=execution_time,
                        success=False,
                        metadata={"error": str(e)}
                    )

                logger.error(f"Role {detected_role} failed: {e}")
                raise

        return wrapper
    return decorator


def track_adaptation(
    adaptation_type: str = "unknown"
) -> Callable[[F], F]:
    """
    Decorator for adaptation process methods.

    Args:
        adaptation_type: Type of adaptation (e.g., "offline", "online", "step")
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            integration = get_integration()
            if not integration.is_available():
                return func(*args, **kwargs)

            try:
                # Import here to avoid circular imports
                from .opik_integration import track, opik_context

                @track(
                    name=f"adaptation_{adaptation_type}_{func.__name__}",
                    tags=["ace-framework", "adaptation", f"adaptation-{adaptation_type}"]
                )
                def traced_adaptation():
                    result = func(*args, **kwargs)

                    # If result contains adaptation metrics, log them
                    if hasattr(result, '__dict__'):
                        result_dict = result.__dict__
                        if 'epoch' in result_dict and 'step' in result_dict:
                            integration.log_adaptation_metrics(
                                epoch=result_dict.get('epoch', 0),
                                step=result_dict.get('step', 0),
                                performance_score=result_dict.get('performance_score', 0.0),
                                bullet_count=result_dict.get('bullet_count', 0),
                                successful_predictions=result_dict.get('successful_predictions', 0),
                                total_predictions=result_dict.get('total_predictions', 0)
                            )

                    return result

                return traced_adaptation()

            except Exception as e:
                logger.warning(f"Failed to trace adaptation {func.__name__}: {e}")
                return func(*args, **kwargs)

        return wrapper
    return decorator


def track_playbook_operation(
    operation_type: Optional[str] = None
) -> Callable[[F], F]:
    """
    Decorator for playbook operations (add, update, remove bullets).

    Args:
        operation_type: Type of operation (auto-detected if not provided)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            integration = get_integration()
            if not integration.is_available():
                return func(*args, **kwargs)

            detected_operation = operation_type or func.__name__

            try:
                # Import here to avoid circular imports
                from .opik_integration import track, opik_context

                @track(
                    name=f"playbook_{detected_operation}",
                    tags=["ace-framework", "playbook", f"playbook-{detected_operation}"]
                )
                def traced_playbook():
                    result = func(*args, **kwargs)

                    # Try to extract playbook metrics from result or instance
                    playbook_instance = None
                    if args and hasattr(args[0], 'bullets'):
                        playbook_instance = args[0]

                    if playbook_instance:
                        total_bullets = len(playbook_instance.bullets)
                        integration.log_playbook_update(
                            operation_type=detected_operation,
                            total_bullets=total_bullets,
                            metadata={"method": func.__name__}
                        )

                    return result

                return traced_playbook()

            except Exception as e:
                logger.warning(f"Failed to trace playbook operation {func.__name__}: {e}")
                return func(*args, **kwargs)

        return wrapper
    return decorator


def track_bullet_evolution(func: F) -> F:
    """
    Decorator specifically for bullet evolution tracking.
    Automatically extracts bullet metrics and logs to Opik.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        integration = get_integration()
        if not integration.is_available():
            return func(*args, **kwargs)

        try:
            # Import here to avoid circular imports
            from .opik_integration import track, opik_context

            @track(
                name=f"bullet_evolution_{func.__name__}",
                tags=["ace-framework", "bullet-evolution"]
            )
            def traced_bullet():
                result = func(*args, **kwargs)

                # Try to extract bullet information from args/kwargs
                if 'bullet_id' in kwargs:
                    bullet_id = kwargs['bullet_id']
                    bullet_content = kwargs.get('bullet_content', '')
                    helpful_count = kwargs.get('helpful_count', 0)
                    harmful_count = kwargs.get('harmful_count', 0)
                    neutral_count = kwargs.get('neutral_count', 0)
                    section = kwargs.get('section', 'unknown')

                    integration.log_bullet_evolution(
                        bullet_id=bullet_id,
                        bullet_content=bullet_content,
                        helpful_count=helpful_count,
                        harmful_count=harmful_count,
                        neutral_count=neutral_count,
                        section=section
                    )

                return result

            return traced_bullet()

        except Exception as e:
            logger.warning(f"Failed to trace bullet evolution {func.__name__}: {e}")
            return func(*args, **kwargs)

    return wrapper
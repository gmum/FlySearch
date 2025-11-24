# Submit to FlySearch benchmark

We invite you to participate in the FlySearch challenge!
Both pure VLM/MLLMs and full agentic frameworks are accepted.

If you are interested in submitting your agent to the [FlySearch leaderboard](../index.md#leaderboard), please do the following:

### Simple VLM/MLLM API based solutions

1. Open a new issue in the [gmum/FlySearch](https://github.com/gmum/FlySearch) repository.
2. In the issue description include the following metadata:
    - `name`: name of your method to be included in the leaderboard,
    - `oss`: `true` iff your agent is open-source (model + strategy),
    - `site`: URL/link to information about your agent (paper/pre-print/blog/repo; optional),
    - `date`: date of the experiment,
    - `logs`: link to a zip file with logs of FlySearch benchmark of you agent (default log location:
      `all_logs/<run-name>` directory).
3. Include instructions on how to reproduce your results. If your approach requires code changes consider opening a pull-request as in the next section.
4. We will verify your submission is possible by running it both on public FlySearch set, and confirm the results on
   separate secret testing set.

Finally, if for some reason (e.g. intellectual property protection) it is not feasible to let us and others
reproduce your results we may still add your agent to the leaderboard with an `unverified` annotation. However, we
strongly encourage that even for non-public solution an API access to the agent is provided for verification (can be privately shared with us by
email).

### Custom models and agentic frameworks

1. Fork and clone the [gmum/FlySearch](https://github.com/gmum/FlySearch) repository.
2. Make necessary code changes, remember to document how to run your version. Keep modifications to the minimum - it
   will be simpler for us to review.
3. Commit the changes and open a new pull request, include all the information as in the simple VLM section above.
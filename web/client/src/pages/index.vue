<template>
  <v-app>
    <v-app-bar
      elevation="1"
    >
      <v-toolbar-title>FlySearch demo</v-toolbar-title>
      <v-spacer/>
      <v-text-field
        v-model="client_name"
        label="Your name"
        variant="underlined"
        class="mr-7 mt-5"
        density="compact"
        max-width="250"
        :disabled="connected"
      />
      <v-select
        v-model="selected_mode"
        :items="modes"
        label="Mode"
        variant="underlined"
        class="mr-7 mt-5"
        density="compact"
        max-width="100"
      />
      <v-btn
        variant="outlined"
        color="error"
        class="mr-5"
        :loading="connecting"
        :disabled="!client_name || client_name.length < 3"
        @click="resetEnv"
      >
        <span v-if="connected">Start a new game</span>
        <span v-else>Connect to server</span>
      </v-btn>
    </v-app-bar>
    <v-main>
      <v-container>
        <v-card
          v-if="!started"
          variant="outlined"
        >
          <v-card-title>Your task</v-card-title>
          <v-card-subtitle>A similar prompt is given to the AI agent</v-card-subtitle>
          <v-card-text>
            <v-list>
              <v-list-item>You are in command of a UAV, tasked with finding a target.</v-list-item>
              <v-list-item>
                You should fly BELOW 10 meters above the object and then reply with "FOUND". Being lower
                (closer to
                the
                object) (like 9, 8, or less meters) is good, being higher than that (like 11, 12, or more meters) is
                bad.
              </v-list-item>
              <v-list-item>
                You may not be able to see the object in the first image, so you need to perform a careful
                search.
                Your
                performance will be evaluated based on whether the object was at most 10 meters below the drone when you
                replied with "FOUND". The object MUST be in your field of view when you reply with "FOUND". You must be
                centered on the object.
              </v-list-item>

              <v-list-item>
                There is a grid overlaid on each image you are presented with. It is meant to (roughly)
                communicate
                which
                point will be in drone's center of vision if you move in that direction. Note that height of the drone
                is
                not represented in the grid.
              </v-list-item>


              <v-list-item>
                To move the drone in a certain direction, use the following format: (x, y, z). For example,
                if you want
                to
                fly to the place denoted as (10, 10) on the grid without changing the altitude, you should reply with
                (10,
                10, 0).
              </v-list-item>

              <v-list-item>
                x and y are the coordinates on the grid, and z is the altitude difference. For example, (0,
                0, -10)
                means
                that you are moving 10 meters down. This is especially important, since you need to get close to the
                object in question. tag should contain the move you are making.
              </v-list-item>

              <v-list-item>
                If you find the object, fly below 10 meters relative to it and reply with "FOUND". Remember,
                it must be in
                your field of view when you reply with "FOUND" and you must be 10 meters above it or closer. Being too
                far
                away is not acceptable.
              </v-list-item>

              <v-list-item>
                You shouldn't move into coordinates that are outside of your view. Otherwise, you may hit
                something which
                is
                not ideal.
                You can make a limited number of moves, dependent on the scenario. Your altitude cannot exceed 300
                meters.
              </v-list-item>

              <v-list-item>
                The search area is limited to what would be visible from the starting position if there were no
                buildings or obstacles. The object is within this area. You may not fly outside of it.
              </v-list-item>
            </v-list>

            <v-alert
              variant="text"
              color="primary"
            >
              To start a new game fill in your username and press the connect button in the top right corner.
            </v-alert>
          </v-card-text>
        </v-card>
        <v-card
          v-if="started && won === null"
          variant="outlined"
        >
          <v-card-title>Target: {{ target }}</v-card-title>
          <v-card-subtitle>
            Move the UAV to locate the target, then click FOUND when the target is visible and altitude
            is less than 10m above target
          </v-card-subtitle>
          <v-card-text>
            <div class="d-flex flex-column align-center mt-1 mb-10">
              <div class="d-flex flex-row align-center justify-center">
                <div class="text-center">
                  <div v-if="image_b64">
                    Camera view
                  </div>
                  <v-img
                    :width="500"
                    :height="500"
                    :src="'data:image/jpeg;base64,' + image_b64"
                    class="mb-2"
                  >
                    <template #placeholder>
                      <div class="d-flex flex-column align-center justify-center fill-height">
                        <div>
                          <v-progress-circular
                            color="grey"
                            indeterminate
                            size="100"
                          />
                        </div>
                      </div>
                    </template>
                  </v-img>
                </div>
                <div
                  v-if="loaded_mode === modes[1]"
                  class="text-center ml-6"
                >
                  <div v-if="image_b64">
                    What the target looks like approximately.
                    <br>
                    May differ slightly,<br>or have multiple different variations.
                  </div>
                  <v-img
                    v-if="image_b64"
                    :width="300"
                    :height="300"
                    :src="'/targets/' + target_image + '.png.jpg'"
                  />
                </div>
              </div>
              <div>Current altitude: {{ altitude }}</div>
              <div :class="collision === true ? 'text-red' : ''">
                Collided on the last action: {{ collision }}
              </div>
              <div :class="moves_left <= 1 ? 'text-red' : ''">
                Remaining moves: {{ moves_left }}
              </div>
              <div :class="status === 'ok' || status === null ? '' : 'text-red'">
                Status: {{ status }}
              </div>
            </div>

            <v-form>
              <v-row>
                <v-col
                  cols="12"
                  sm="12"
                  md="5"
                  class="text-center"
                >
                  <div>If the object is visible and the altitude is less than 10m above the target</div>
                  <v-btn
                    variant="outlined"
                    :disabled="!image_b64"
                    class="mt-3"
                    color="success"
                    @click="action(true)"
                  >
                    Found
                  </v-btn>
                </v-col>
                <v-col
                  cols="12"
                  sm="12"
                  md="2"
                  class="text-center"
                >
                  <v-divider
                    vertical
                    class=""
                  >
                    Or
                  </v-divider>
                </v-col>
                <v-col
                  cols="12"
                  sm="12"
                  md="5"
                  class="text-center"
                >
                  <div class="text-center">
                    Else, move the UAV closer to the target
                  </div>
                  <div class="d-flex flex-row justify-center align-center mt-2">
                    <v-text-field
                      v-model="x"
                      :disabled="!image_b64"
                      variant="outlined"
                      type="number"
                      label="X"
                      control-variant="stacked"
                      class="mr-2"
                      precision="0"
                    />
                    <v-text-field
                      v-model="y"
                      :disabled="!image_b64"
                      variant="outlined"
                      hide-spin-buttons
                      type="number"
                      label="Y"
                      control-variant="stacked"
                      class="mr-2"
                      precision="0"
                    />
                    <v-text-field
                      v-model="z"
                      :disabled="!image_b64"
                      variant="outlined"
                      hide-spin-buttons
                      type="number"
                      label="Z"
                      control-variant="stacked"
                      class="mr-2"
                      precision="0"
                    />
                  </div>
                  <v-btn
                    :disabled="!image_b64"
                    variant="outlined"
                    color="primary"
                    class="mt-n2"
                    @click="action(false)"
                  >
                    Move
                  </v-btn>
                </v-col>
              </v-row>
            </v-form>
          </v-card-text>
        </v-card>
        <v-card
          v-if="won !== null"
          variant="outlined"
        >
          <v-card-title>{{ won ? "You won!" : "You lost!" }}</v-card-title>
          <v-card-text>
            <v-alert
              v-if="won === true"
              variant="outlined"
              color="success"
            >
              You've found the target. To start a new game press the button in the top right corner.
            </v-alert>
            <v-alert
              v-if="won === false"
              variant="outlined"
              color="error"
            >
              You've failed to locate the target. To start a new game press the button in the top right corner.
            </v-alert>
            <v-alert
              v-if="last_standardised_scenario === true"
              variant="outlined"
              color="primary"
              class="mt-4"
            >
              There are no more benchmark scenarios available. You can still play the game with random scenarios or you
              can switch to other modes.
            </v-alert>
          </v-card-text>
        </v-card>
      </v-container>
    </v-main>
    <v-dialog
      v-model="error"
      class="align-center justify-center"
      persistent
      max-width="500"
    >
      <v-card
        prepend-icon="mdi-alert-circle"
        :value="error"
        title="Error occurred!"
        min-width="500"
      >
        <v-card-text>
          <div>{{ error_message }}</div>
          <div class="mt-4">
            Try to reload the app in 3 minutes or contact the developers if the problem persists.
          </div>
        </v-card-text>
        <v-card-actions>
          <v-btn
            variant="text"
            color="error"
            @click="reload"
          >
            Reload app
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
    <v-fab
      app
      color="surface-variant"
      extended
      prepend-icon="mdi-bug"
      text="Report issue"
      variant="outlined"
      :disabled="!connected"
      @click="report_issue = true"
    />
    <v-dialog
      v-model="report_issue"
      class="align-center justify-center"
      max-width="800"
    >
      <v-card>
        <v-toolbar>
          <v-btn
            icon="mdi-close"
            @click="report_issue = false"
          />
          <v-toolbar-title>Report an issue</v-toolbar-title>
        </v-toolbar>
        <v-card-text>
          <v-textarea
            v-model="issue_description"
            label="Description"
            variant="outlined"
            :disabled="report_sending"
          />
        </v-card-text>
        <v-card-actions>
          <v-btn
            variant="text"
            color="primary"
            :loading="report_sending"
            @click="send_report"
          >
            Send
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-app>
</template>

<script setup>
import {inject, ref} from "vue";

const axios = inject('axios');

const api_base = '/api';
const modes = ['FS-1', 'FS-2']
const selected_mode = ref(modes[0]);
const loaded_mode = ref(modes[0]);

const client_uuid = crypto.randomUUID().toString();
const client_name = ref(null);
const connected = ref(false);
const connecting = ref(false);

const x = ref(0);
const y = ref(0);
const z = ref(0);
const image_b64 = ref(null);
const collision = ref(null);
const altitude = ref(null);
const target = ref(null);
const target_image = ref('');
const started = ref(false);
const won = ref(null);
const moves_left = ref(null);
const status = ref(null);
const last_standardised_scenario = ref(false);

const error = ref(false);
const error_message = ref('');
const report_issue = ref(false);
const issue_description = ref('');
const report_sending = ref(false);

function reload() {
  window.location.reload();
}

function send_report() {
  report_sending.value = true;
  axios.post(api_base + '/complain', {
    'report': issue_description.value,
  }, {params: {'client_uuid': client_uuid}})
    .then(() => {
      report_issue.value = false;
      report_sending.value = false;
      issue_description.value = '';
    })
    .catch((err) => {
      error.value = true;
      error_message.value = err.message;
      console.error(err);
    });
}

function cleanStatus() {
  image_b64.value = null;
  collision.value = null;
  altitude.value = null;
}

function ping(keepalive = false) {
  if (!connected.value) {
    connecting.value = true;
  }

  axios.post(api_base + '/ping', {
    'client_uuid': client_uuid,
    'client_name': client_name.value,
  })
    .then(() => {
      setTimeout(() => ping(keepalive = true), 1000);
      if (!keepalive) {
        connecting.value = false;
        connected.value = true;
        started.value = true;
        resetEnv();
      }
    })
    .catch((err) => {
      if (err.response && err.response.status === 409) {
        error.value = true;
        error_message.value = 'Someone else is already using the server. Please try again later.';
        return;
      }
      error.value = true;
      error_message.value = err.message;
      console.error(err);
    });
}

function getStatus() {
  axios.get(api_base + '/get_observation', {params: {'client_uuid': client_uuid}})
    .then((response) => {
      const current_status = response.data;
      image_b64.value = current_status.image_b64;
      collision.value = current_status.collision;
      altitude.value = current_status.altitude;
      x.value = 0
      y.value = 0
      z.value = 0
      loaded_mode.value = selected_mode.value;
    })
    .catch((err) => {
      error.value = true;
      error_message.value = err.message;
      console.error(err);
    });
}

function resetEnv() {
  if (!connected.value) {
    ping()
    return;
  }

  started.value = true;
  won.value = null;
  last_standardised_scenario.value = false;
  cleanStatus();
  target.value = null;
  const request = {
    is_fs1: selected_mode.value === modes[0],
  };
  axios.post(api_base + '/generate_new', request, {params: {'client_uuid': client_uuid}})
    .then((response) => {
      const data = response.data
      target.value = data.target;
      target_image.value = data.target.replace('a ', '').replaceAll(' ', '_');
      moves_left.value = data.moves_left;
      status.value = 'ok';
      getStatus();
    })
    .catch((err) => {
      error.value = true;
      error_message.value = err.message;
      console.error(err);
    });
}

function action(is_done) {
  cleanStatus();

  const x1 = parseInt(x.value) || 0;
  const y1 = parseInt(y.value) || 0;
  const z1 = parseInt(z.value) || 0;

  const request = {
    found: is_done,
    coordinate_change: is_done ? [0, 0, 0] : [x1, y1, z1]
  };

  axios.post(api_base + '/move', request, {params: {'client_uuid': client_uuid}})
    .then((response) => {
      const data = response.data;
      if (is_done) {
        won.value = data.success;
        last_standardised_scenario.value = data.last_standardised_scenario;
      } else {
        moves_left.value = data.moves_left;
        status.value = 'ok';

        if (moves_left.value === 0) {
          won.value = false;
          last_standardised_scenario.value = data.last_standardised_scenario;
        } else {
          getStatus();
        }
      }
    })
    .catch((err) => {
      if (err.response && err.response.data && err.response.data.user_error) {
        status.value = err.response.data.user_error;
        getStatus();
        return;
      }
      error.value = true;
      error_message.value = err.message;
      console.error(err);
    });
}
</script>

<!--suppress CssUnusedSymbol -->
<style scoped>
.v-container {
  max-width: 1000px;
  margin-top: 1rem
}
</style>
